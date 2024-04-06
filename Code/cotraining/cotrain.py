from VMDNet.code.train import VMDTrainer
from HetNet.train import HetNetTrainer
import uuid
import os
import torch

pretrain = False # Set this to false if the pretraining has already been done and the model has been saved. WILL ONLY PRETRAIN - WON'T COTRAIN.
pretrained_vmd_model_path = "/home/s2062378/cotraining/model/vmdnetpretrained.pth"
pretrained_hetnet_model_path = "/home/s2062378/cotraining/model/hetnetpretrained.pth"

vmd_args = {
    'batch_size': 6,
    'self_training_start_epoch': 5,
    'self_training_confidence_threshold': 0.65,
    'data_loader_workers': 0,
    'finetune_lr': 1e-4,
    'scratch_lr': 1e-3,
    'weight_decay': 5e-4,
    'scale': 416,
    'warm_up_epochs': 3,
    'seed': 2024
}

hetnet_args = {
    'batch_size': 6,
    'lr': 0.005,
    'momentum': 0.9,
    'decay': 5e-4,
    'max_epoch': 150,
    'num_workers': 4, # If doesn't work, turn back down to 3.
    'random_seed': 7,
    'self_training_start_epoch': 75,
}

print("\nVMDNet arguments: \n", vmd_args)
print("\nHetNet arguments: \n", hetnet_args)

unlabeled_dataset = "/exports/eddie/scratch/s2062378/dataset/VMD/unlabeled"

VMDNet_combined_dataset_dir_name = "/exports/eddie/scratch/s2062378/dataset/VMD/VMD_combined_dataset" + str(uuid.uuid4())
HetNet_combined_dataset_root = "/exports/eddie/scratch/s2062378/dataset/VMD/HETNET_combined_dataset" + str(uuid.uuid4())
HetNet_combined_dataset_dir_name = HetNet_combined_dataset_root + "/train/"

# Make the directories
os.makedirs(VMDNet_combined_dataset_dir_name, exist_ok=True)
os.makedirs(HetNet_combined_dataset_dir_name, exist_ok=True)

# Copy all files from the training set into the combined datasets
os.system(f"cp -r {os.path.join('/exports/eddie/scratch/s2062378/dataset/VMD/train', '*')} {os.path.join(VMDNet_combined_dataset_dir_name)}")
os.system(f"cp -r {os.path.join('/exports/eddie/scratch/s2062378/dataset/VMD/train', '*')} {os.path.join(HetNet_combined_dataset_dir_name)}")

if pretrain:
    print("Training VMDNet before the co-training starts...")
    vmd_trainer = VMDTrainer(vmd_args)
    vmd_trainer.train(vmd_args['self_training_start_epoch'], [VMDNet_combined_dataset_dir_name, 'video', 'VMD_train'])
    vmd_trainer.save_model(pretrained_vmd_model_path)

    torch.cuda.empty_cache() # Clear cache

    print("Training HetNet before the co-training starts...")
    hetnet_trainer = HetNetTrainer(hetnet_args, HetNet_combined_dataset_root)
    hetnet_trainer.train(hetnet_args['self_training_start_epoch'])
    hetnet_trainer.save_model(pretrained_hetnet_model_path)
    exit()
    
    
else:
    vmd_trainer = VMDTrainer(vmd_args)
    vmd_trainer.load_model(pretrained_vmd_model_path)
    hetnet_trainer = HetNetTrainer(hetnet_args, HetNet_combined_dataset_root)
    hetnet_trainer.load_model(pretrained_hetnet_model_path)

print("Starting co-training...")
for cotrain_epoch in range(5):
    # Use VMDNet to label unlabeled and put confident predictions in the HetNet dataset
    vmd_trainer.label_unlabeled(unlabeled_dataset, HetNet_combined_dataset_dir_name)
    
    # Use HetNet to label unlabeled and put confident predictions in the VMDNet dataset
    confidences = hetnet_trainer.label_unlabeled("/exports/eddie/scratch/s2062378/dataset/VMD", "/exports/eddie/scratch/s2062378/dataset/VMD/temp", 0)
    
    # Calculate the top 10% of the confidences
    confidences = {k: v for k, v in sorted(confidences.items(), key=lambda item: item[1], reverse=True)[:int(len(confidences)) * 0.1]}
    
    for video in confidences:
        os.system(f"cp -r {os.path.join('/exports/eddie/scratch/s2062378/dataset/VMD/temp', video)} {os.path.join(VMDNet_combined_dataset_dir_name, video)}")
            
    # Remove the temporary directory
    os.system(f"rm -r /exports/eddie/scratch/s2062378/dataset/VMD/temp")
    
    torch.cuda.empty_cache() # Clear cache
    
    # Train VMDNet
    vmd_trainer.train(1, [VMDNet_combined_dataset_dir_name, 'video', 'VMD_train'])
    torch.cuda.empty_cache() # Clear cache
    
    # Train HetNet
    hetnet_trainer.train(15)
    torch.cuda.empty_cache() # Clear cache
    
vmd_trainer.save_model("/home/s2062378/cotraining/")

# Remove the temporary combined datasets
os.system(f"rm -r {VMDNet_combined_dataset_dir_name}")
os.system(f"rm -r {HetNet_combined_dataset_dir_name}")