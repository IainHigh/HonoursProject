#!/usr/bin/env python3

import os
import math
import torch.nn.functional
import numpy as np
import importlib
import torch
from PIL import Image
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from VMDNet.code import joint_transforms
from VMDNet.code.losses import lovasz_hinge
from VMDNet.code.dataset.VShadow_crosspairwise import CrossPairwiseImg
from VMDNet.code.misc import AvgMeter, check_mkdir
from VMDNet.code.networks.VMD_network import VMD_Network
from VMDNet.code.dataset.VShadow_crosspairwise import listdirs_only

# Note to self:
# query_index is one after exemplar_index (query index is the one we are trying to predict, exemplar index is the one we are trying to predict it from)

class VMDTrainer:
    def __init__(self, args):
        self.args = args
        VMD_Network = importlib.import_module('VMDNet.code.networks.VMD_network').VMD_Network
        self.model = torch.nn.DataParallel(VMD_Network()).cuda().train()
        
        cudnn.deterministic = True
        cudnn.benchmark = False

        # fix random seed
        np.random.seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        
        self.params = [
                {"params": [param for name, param in self.model.named_parameters() if 'backbone' in name], "lr": args['finetune_lr']},
                {"params": [param for name, param in self.model.named_parameters() if 'backbone' not in name], "lr": args['scratch_lr']},
            ]
        self.optimizer = optim.Adam(self.params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
        
        warm_up_with_cosine_lr = lambda epoch: epoch / args['warm_up_epochs'] if epoch <= args['warm_up_epochs'] else 0.5 * \
                             (math.cos((epoch-args['warm_up_epochs'])/(10-args['warm_up_epochs'])*math.pi)+1)    
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_cosine_lr)


    def label_unlabeled(self, input_dir, output_dir):
        img_transform = transforms.Compose([
            transforms.Resize((self.args['scale'], self.args['scale'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        to_pil = transforms.ToPILImage()
        self.model.eval()
        video_list = listdirs_only(os.path.join(input_dir))
        
        check_mkdir(output_dir)
        
        for video in tqdm(video_list, desc="VMDNet - Labelling unannotated dataset"):
            self.process_video_batch((video, input_dir, output_dir, img_transform, to_pil))

    def process_video_batch(self, func_args, batch_size=256):
        video, input_dir, output_dir, img_transform, to_pil = func_args
        with torch.no_grad():
            total_confidence = 0
            check_mkdir(os.path.join(input_dir, video, "SegmentationClassPNG"))
            img_list = self.sortImg([os.path.splitext(f)[0] for f in os.listdir(os.path.join(input_dir, video, "JPEGImages")) if f.endswith('.jpg')])

            exemplar_tensors, query_tensors, img_sizes = [], [], []
            for exemplar_idx, exemplar_name in enumerate(img_list):
                query_idx = self.getAdjacentIndex(exemplar_idx, 0, len(img_list))
                exemplar = Image.open(os.path.join(input_dir, video, "JPEGImages", exemplar_name + '.jpg')).convert('RGB')
                query = Image.open(os.path.join(input_dir, video, "JPEGImages", img_list[query_idx] + '.jpg')).convert('RGB')
                w, h = exemplar.size
                img_sizes.append((h, w))
                exemplar_tensors.append(img_transform(exemplar).unsqueeze(0))
                query_tensors.append(img_transform(query).unsqueeze(0))

            # Process in batches
            for i in range(0, len(exemplar_tensors), batch_size):
                torch.cuda.empty_cache() # Clear cache
                batch_exemplar_tensors = torch.cat(exemplar_tensors[i:i+batch_size]).cuda()
                batch_query_tensors = torch.cat(query_tensors[i:i+batch_size]).cuda()
                exemplar_pre, _ = self.model(batch_exemplar_tensors, batch_query_tensors)

                for j, res in enumerate(exemplar_pre):
                    total_confidence += torch.mean((2 / (1 + torch.exp(-torch.abs(res)))) - 1).item()
                    res = (res.data > 0).to(torch.float32)
                    prediction = np.array(transforms.Resize(img_sizes[i+j])(to_pil(res.cpu())))
                    save_name = f"{img_list[i+j]}.png"
                    Image.fromarray(prediction).save(os.path.join(input_dir, video, "SegmentationClassPNG", save_name))

            average_confidence = total_confidence / len(img_list)
            new_vid_name = video + "_labeled"
            os.system("rm -rf %s" % (os.path.join(output_dir, new_vid_name))) # Remove existing one if it exists
            if average_confidence > self.args['self_training_confidence_threshold']:   
                os.system("cp -rf %s %s" % (os.path.join(input_dir, video), os.path.join(output_dir, new_vid_name)))

    def sortImg(self, img_list):
        img_int_list = [int(f.split(".")[0]) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]

    def getAdjacentIndex(self, current_index, start_index, video_length):
        if current_index + 1 < start_index + video_length:
            return current_index+1
        return current_index-1

    def train(self, num_epochs, training_root):
        self.model.train()
        target_transform = transforms.ToTensor()
        
        img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.Resize((self.args['scale'], self.args['scale']))
        ])
        
        train_set = CrossPairwiseImg([training_root], joint_transform, img_transform, target_transform)
        train_loader = DataLoader(train_set, batch_size=self.args['batch_size'], drop_last=True, num_workers=self.args['data_loader_workers'], shuffle=True)

        for cur_epoch in range(num_epochs):
            loss_record1, loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
            loss_record5, loss_record6, loss_record7 = AvgMeter(), AvgMeter(), AvgMeter()
            train_iterator = tqdm(train_loader, total=len(train_loader), desc="VMDNet - training")
            torch.cuda.empty_cache()

            for _, sample in enumerate(train_iterator):
                
                exemplar, exemplar_gt, query, query_gt = sample['exemplar'].cuda(), sample['exemplar_gt'].cuda(), sample['query'].cuda(), sample['query_gt'].cuda()

                self.optimizer.zero_grad()
                exemplar_pre, query_pre, examplar_final, query_final, = self.model(exemplar, query)
                
                loss_hinge1 = lovasz_hinge(exemplar_pre, exemplar_gt)
                loss_hinge2 = lovasz_hinge(query_pre, query_gt)
                loss_hinge_examplar = lovasz_hinge(examplar_final, exemplar_gt)
                loss_hinge_query = lovasz_hinge(query_final, query_gt)

                # classification loss
                loss = loss_hinge1 + loss_hinge2 + loss_hinge_examplar + loss_hinge_query
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)  # gradient clip
                self.optimizer.step()  # change gradient
                loss_record1.update(loss_hinge_examplar.item(), self.args['batch_size'])
                loss_record2.update(loss_hinge_query.item(), self.args['batch_size'])
                loss_record4.update(loss_hinge1.item(), self.args['batch_size'])
                loss_record5.update(loss_hinge2.item(), self.args['batch_size'])
            self.scheduler.step()  # change learning rate after epoch
    
    def save_model(self, path):
        checkpoint = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        torch.save(checkpoint, path)
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])