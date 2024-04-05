#!/bin/bash
#
#$ -N Original
#
# Runtime limit of 48 hours:
#$ -l h_rt=48:00:00
#
# Set working directory to the directory to VMD-Net
#$ -wd /home/s2062378/VMD-Net_Original/
# Set the output and error stream to the output directory:
#$ -o /home/s2062378/output
#$ -e /home/s2062378/output
#
# Send email when the program (b) begins, (e) ends, (a) aborts, (s) suspends.
#$ -m beas
#$ -M iain.high@sky.com
#
# Request two GPUs in the gpu queue:
#$ -q gpu 
#$ -pe gpu-a100 1
#
# Request 80 GB system RAM per GPU
#$ -l h_vmem=80G
#
# Request resource reservation
#$ -R y

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda

# Load Python through Anaconda
module load anaconda
source activate new_env

echo -n $CUDA_VISIBLE_DEVICES

# Train the model
# python3 /home/s2062378/VMD-Net_Original/code/train.py

# Run the test set through the model
#python3 /home/s2062378/VMD-Net_Original/code/infer.py

# Evaluate the model on the test set
python3 /home/s2062378/VMD-Net_Original/code/eval.py

# Infer and run on Pexels annotated images

# Run the test set through the model
# python3 /home/s2062378/VMD-Net_Original/code/infer_pexels.py

# Evaluate the model on the test set
# python3 /home/s2062378/VMD-Net_Original/code/eval_pexels.py