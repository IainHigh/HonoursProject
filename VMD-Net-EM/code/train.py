#!/usr/bin/env python3

import os
import joint_transforms
import math
import torch.nn.functional
import numpy as np
import importlib
import torch

import uuid
from PIL import Image
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from losses import lovasz_hinge
from config import VMD_training_root, VMD_Unlabeled_root
from dataset.VShadow_crosspairwise import CrossPairwiseImg
from misc import AvgMeter, check_mkdir
from networks.VMD_network import VMD_Network
from dataset.VShadow_crosspairwise import listdirs_only

# Note to self:
# query_index is one after exemplar_index (query index is the one we are trying to predict, exemplar index is the one we are trying to predict it from)

cudnn.deterministic = True
cudnn.benchmark = False

ckpt_path = './model'
VMD_Network = importlib.import_module('networks.VMD_network').VMD_Network

combined_dataset_dir_name = "/exports/eddie/scratch/s2062378/dataset/VMD/temp_combined_dataset_EM" + str(uuid.uuid4()) # Create a random directory name to avoid conflict

args = {
    'max_epoch': 10,
    'batch_size': 5,
    'self_training_start_epoch': 5,
    'self_training_confidence_threshold': 0.80, # confidence threshold for EM
    'data_loader_workers': 0,
    'finetune_lr': 1e-4,
    'scratch_lr': 1e-3,
    'weight_decay': 5e-4,
    'scale': 416, # Increasing this increases performance but also increases training time and memory usage. Default value 416. Try 256 and 512. TODO
    'warm_up_epochs': 3,
    'seed': 2024
}

# Testing with variable batchsize depending on maxVMem, and number of videos on the iteration.

# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

net = torch.nn.DataParallel(VMD_Network()).cuda().train()

def main():
    # multi-GPUs training    
    params = [
        {"params": [param for name, param in net.named_parameters() if 'backbone' in name], "lr": args['finetune_lr']},
        {"params": [param for name, param in net.named_parameters() if 'backbone' not in name], "lr": args['scratch_lr']},
    ]
    optimizer = optim.Adam(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    warm_up_with_cosine_lr = lambda epoch: epoch / args['warm_up_epochs'] if epoch <= args['warm_up_epochs'] else 0.5 * \
                             (math.cos((epoch-args['warm_up_epochs'])/(args['max_epoch']-args['warm_up_epochs'])*math.pi)+1)    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    check_mkdir(ckpt_path)
    
    check_mkdir(os.path.join(combined_dataset_dir_name))
    # Copy all the images from the labeled dataset to the combined dataset
    os.system(f"cp -r {os.path.join([VMD_training_root][0][0], '*')} {os.path.join(combined_dataset_dir_name)}")
    
    print('\n' + str(args) + '\n')
    train(optimizer, scheduler)


def label_unlabeled(input_dir):
    img_transform = transforms.Compose([
        transforms.Resize((args['scale'], args['scale'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    to_pil = transforms.ToPILImage()
    net.eval()
    video_list = listdirs_only(os.path.join(input_dir))
    for video in tqdm(video_list, desc="Labelling unannotated dataset"):
        process_video_batch((video, input_dir, combined_dataset_dir_name, img_transform, to_pil))

def process_video_batch(func_args, batch_size=256):
    video, input_dir, combined_dataset_dir_name, img_transform, to_pil = func_args
    with torch.no_grad():
        total_confidence = 0
        check_mkdir(os.path.join(input_dir, video, "SegmentationClassPNG"))
        img_list = sortImg([os.path.splitext(f)[0] for f in os.listdir(os.path.join(input_dir, video, "JPEGImages")) if f.endswith('.jpg')])

        exemplar_tensors, query_tensors, img_sizes = [], [], []
        for exemplar_idx, exemplar_name in enumerate(img_list):
            query_idx = getAdjacentIndex(exemplar_idx, 0, len(img_list))
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
            exemplar_pre, _ = net(batch_exemplar_tensors, batch_query_tensors)

            for j, res in enumerate(exemplar_pre):
                total_confidence += torch.mean((2 / (1 + torch.exp(-torch.abs(res)))) - 1).item()
                res = (res.data > 0).to(torch.float32)
                prediction = np.array(transforms.Resize(img_sizes[i+j])(to_pil(res.cpu())))
                save_name = f"{img_list[i+j]}.png"
                Image.fromarray(prediction).save(os.path.join(input_dir, video, "SegmentationClassPNG", save_name))

        average_confidence = total_confidence / len(img_list)
        old_vid_name = video + "_labeled_" + "*"
        new_vid_name = video + "_labeled_" + str(average_confidence)
        os.system("rm -rf %s" % (os.path.join(combined_dataset_dir_name, old_vid_name))) # Remove existing one if it exists
        if average_confidence > args['self_training_confidence_threshold']:   
            os.system("cp -rf %s %s" % (os.path.join(input_dir, video), os.path.join(combined_dataset_dir_name, new_vid_name)))

def sortImg(img_list):
    img_int_list = [int(f.split(".")[0]) for f in img_list]
    sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
    return [img_list[i] for i in sort_index]

def getAdjacentIndex(current_index, start_index, video_length):
    if current_index + 1 < start_index + video_length:
        return current_index+1
    return current_index-1

def train(optimizer, scheduler):
    target_transform = transforms.ToTensor()
    curr_epoch = 1
    curr_iter = 1
    
    img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
    ])
    print("Expectation Maximisation (EM)")
    print("=====>Dataset loading<======")
    training_root = [VMD_training_root] # training_root should be a list form, like [datasetA, datasetB, datasetC], here we use only one dataset.
    train_set = CrossPairwiseImg(training_root, joint_transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], drop_last=True, num_workers=args['data_loader_workers'], shuffle=True)

    unlabeled_root = [VMD_Unlabeled_root] # unlabeled_root should be a list form, like [datasetA, datasetB, datasetC], here we use only one dataset.

    print('=====>Start training<======')
    while True:
        loss_record1, loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_record5, loss_record6, loss_record7 = AvgMeter(), AvgMeter(), AvgMeter()
        train_iterator = tqdm(train_loader, total=len(train_loader), desc="training epoch %d" % curr_epoch)
        torch.cuda.empty_cache()
        

        for _, sample in enumerate(train_iterator):
            
            exemplar, exemplar_gt, query, query_gt = sample['exemplar'].cuda(), sample['exemplar_gt'].cuda(), sample['query'].cuda(), sample['query_gt'].cuda()

            optimizer.zero_grad()
            exemplar_pre, query_pre, examplar_final, query_final, = net(exemplar, query)
            
            loss_hinge1 = lovasz_hinge(exemplar_pre, exemplar_gt)
            loss_hinge2 = lovasz_hinge(query_pre, query_gt)
            loss_hinge_examplar = lovasz_hinge(examplar_final, exemplar_gt)
            loss_hinge_query = lovasz_hinge(query_final, query_gt)

            # classification loss
            loss = loss_hinge1 + loss_hinge2 + loss_hinge_examplar + loss_hinge_query
            
            # Get the confidence of the model
            confidence = sample['confidence'].cuda()
            confidence = confidence[0].item()
            weighted_loss = loss * confidence
            
            weighted_loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 12)  # gradient clip
            optimizer.step()  # change gradient
            loss_record1.update(loss_hinge_examplar.item(), args['batch_size'])
            loss_record2.update(loss_hinge_query.item(), args['batch_size'])
            loss_record4.update(loss_hinge1.item(), args['batch_size'])
            loss_record5.update(loss_hinge2.item(), args['batch_size'])

            curr_iter += 1

        checkpoint = {
            'model': net.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(ckpt_path, 'best_mae.pth'))

        if curr_epoch >= args['max_epoch']:
            return
        
        # Self-Training Iteration (after initial training)
        if curr_epoch >= args['self_training_start_epoch']:
            torch.cuda.empty_cache()
            label_unlabeled(unlabeled_root[0][0]) # Label the unlabeled dataset using the current model, copy the most confident videos to the combined dataset
            torch.cuda.empty_cache()
            train_set = CrossPairwiseImg([(combined_dataset_dir_name, 'video')], joint_transform, img_transform, target_transform)
            train_loader = DataLoader(train_set, batch_size=args['batch_size'], drop_last=True, num_workers=args['data_loader_workers'], shuffle=True)
            net.train() # val -> train
        
        curr_epoch += 1
        scheduler.step()  # change learning rate after epoch

if __name__ == '__main__':
    main()
    # remove combined_dataset_dir_name
    os.system(f"rm -rf {combined_dataset_dir_name}")