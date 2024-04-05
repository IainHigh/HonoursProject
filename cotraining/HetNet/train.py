# coding=utf-8

import sys
import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm
from PIL import Image

import HetNet.dataset as dataset
from HetNet.Net import Net
from HetNet.misc import *

class HetNetTrainer():
    def __init__(self, args, combined_dir_root):
        
        self.args = args
        
        # Create the .txt files listing the images in the VMD dataset
        self.create_samples_list(combined_dir_root + '/train', combined_dir_root + '/train.txt')
                
        
        ## Set random seeds
        seed = self.args['random_seed']
        
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        ## dataset
        # Train the network
        self.cfg = dataset.Config(dataset='VMD', datapath=combined_dir_root, savepath=f'./model/', mode='train', batch=self.args['batch_size'], lr=self.args['lr'], momen=self.args['momentum'], decay=self.args['decay'], epoch=self.args['max_epoch'])
        data = dataset.Data(self.cfg)
        self.loader = DataLoader(data, collate_fn=data.collate, batch_size=self.cfg.batch, shuffle=True, num_workers=self.args['num_workers'])
        
        ## network
        self.net = Net(self.cfg)
        self.net.train(True)
        self.net.cuda()
        
        ## parameter
        enc_params, dec_params = [], []
        for name, param in self.net.named_parameters():
            if 'bkbone' in name:
                enc_params.append(param)
            else:
                dec_params.append(param)

        self.optimizer = torch.optim.SGD([{'params': enc_params}, {'params': dec_params}], lr=self.cfg.lr, momentum=self.cfg.momen, weight_decay=self.cfg.decay, nesterov=True)
        self.scaler = GradScaler()
        self.sw = SummaryWriter(self.cfg.savepath)

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


    def train(self, num_epochs):
        self.net.train(True)
        global_step = 1
        for epoch in range(num_epochs):
            self.optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(num_epochs+1)*2-1))*self.cfg.lr*0.1
            self.optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(num_epochs+1)*2-1))*self.cfg.lr

            for (image, mask) in tqdm(self.loader, desc="HetNet - training"):
                image, mask = image.cuda().float(), mask.cuda().float()
                self.optimizer.zero_grad()

                with autocast():
                    out1, _, out2,  out3, out4, out5 = self.net(image)
                    loss1 = self.structure_loss(out1, mask)
                    loss2 = self.structure_loss(out2, mask)
                    loss3 = self.structure_loss(out3, mask)
                    loss4 = self.structure_loss(out4, mask)
                    loss5 = self.structure_loss(out5, mask)

                    # Aggregate valid losses directly without recreating tensors
                    valid_losses = [loss for loss in [loss1, loss2, loss3, loss4, loss5] if not torch.isnan(loss)]
                    if valid_losses:
                        loss = sum(valid_losses) / len(valid_losses)  # Average or sum, depending on your preference
                    else:
                        print("All losses are NaN. Skipping update.")
                        continue  # Skip this step entirely

                # Scale and backpropagate the loss as usual
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()  # Clear gradients after update

                ## log
                global_step += 1
                self.sw.add_scalar('lr', self.optimizer.param_groups[1]['lr'], global_step=global_step)

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))

    def create_samples_list(self, vmd_dir, output_file):
        """
        Creates a .txt file listing all JPEG images in the VMD directory structure.
        
        Args:
        - vmd_dir (str): Path to the VMD directory.
        - output_file (str): Path to the output .txt file.
        """
        with open(output_file, 'w') as f:
            for root, _, files in os.walk(vmd_dir):
                for file in files:
                    if file.endswith('.jpg') and 'JPEGImages' in root:
                        # Extract the relative path components
                        rel_path = os.path.relpath(root, vmd_dir)
                        video_name = rel_path.split(os.path.sep)[0]
                        image_name = os.path.splitext(file)[0]  # Remove the extension
                        # Write the relative path (video_name/image_name) to the file
                        f.write(f"{video_name}/{image_name}\n")
                        
                        
    def label_unlabeled(self, input_path, output_path, num_workers=0):
        self.net.train(False)
        video_confidence = {}
        video_length = {}
        self.create_samples_list(input_path + '/unlabeled', input_path + '/unlabeled.txt')
        cfg = dataset.Config(dataset='VMD', datapath=input_path, mode='unlabeled')
        data = dataset.Data(cfg)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=num_workers)
        
        with torch.no_grad():
            for image, _, shape, name in tqdm(loader, desc="HetNet - Labelling unannotated dataset"):
                torch.cuda.empty_cache() # Clear cache
                image = image.cuda().float()
                torch.cuda.synchronize()
                out = self.net(image, shape)
                torch.cuda.synchronize()
                pred = torch.sigmoid(out[0]) * 255  # Apply sigmoid and scale to 0-255 range
                
                # Linear confidence calculation
                confidence = torch.mean(torch.abs(pred - 127.5) / 127.5).item()
                
                pred = torch.where(pred >= 127.5, torch.ones_like(pred) * 255, torch.zeros_like(pred)).cpu().numpy()               
                
                pred = np.round(pred).astype(np.uint8)  # Convert to uint8, ensuring binary mask
                
                video_name = name[0].split('/')[0]
                save_path = output_path + "/" + video_name

                # if the video is already in the dictionary, update the total confidence and length
                if video_name in video_confidence:
                    video_confidence[video_name] += confidence
                    video_length[video_name] += 1
                else:
                    video_confidence[video_name] = confidence
                    video_length[video_name] = 1

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                mask_path = save_path+'/'+ name[0].split('/')[1] +'_mask.png'
                Image.fromarray(pred[0, 0]).save(mask_path)  # Save the first channel as a binary mask image
                
        # Calculate the average confidence for each video
        for video in video_confidence:
            video_confidence[video] /= video_length[video]
            print(f'{video}: {video_confidence[video]}')
            
        return video_confidence