#!/bin/bash
#
#$ -N SIMILARITY
#
# Runtime limit of 48 hours:
#$ -l h_rt=48:00:00
#
# Set working directory to the directory to VMD-Net
#$ -wd /home/s2062378/
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
source activate new_env_recent_torch

echo -n $CUDA_VISIBLE_DEVICES

python3 /home/s2062378/similarity.py