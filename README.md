# HonoursProject

Honours project for University of Edinburgh Informatics 2024: Semi-Supervised Approaches to Mirror Detection in Videos

## Setup

1. Clear all conda files and packages (including .cache)
2. Create the anaconda directory and "envs" and "pkgs" sub-directories
3. $ module load anaconda
4. $ conda clean --all
5. $ conda config --add envs_dirs /exports/eddie/scratch/s2062378/anaconda/envs
6. $ conda config --add pkgs_dirs /exports/eddie/scratch/s2062378/anaconda/pkgs
7. $ conda create -n new_env python=3.7 -y
8. $ conda activate new_env
9. $ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y
10. $ conda install pip -y
11. $ pip install tqdm
12. $ pip install SimpleITK===1.2.4
13. $ pip install medpy
14. $ pip install joblib
15. $ pip install opencv-python
16. $ pip install tensorflow[and-cuda] tensorflow_decision_forests
17. $ pip install pandas
18. $ pip install tensorboardX
19. $ pip install scikit-image
20. $ pip install scikit-learn

21. $ conda create -n new_env_recent_torch python=3.7 -y
22. $ conda activate new_env_recent_torch
23. $ conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -y
24. $ conda install pip -y
25. $ pip install tqdm
26. $ pip install matplotlib

## Dataset:

To recreate the dataset, run the code in the pexels-scraper directory. This will download the images and videos from the Pexels API. Before running the code, you will need to create a Pexels API key and store it in the pexel_key.py file.
