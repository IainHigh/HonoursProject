import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as tr
from torchvision.models import vit_h_14
import os
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


class cosineSimilarity:
    def __init__(self, device=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()

    def _load_model(self):
        wt = torchvision.models.ViT_H_14_Weights.DEFAULT
        model = vit_h_14(weights=wt)
        model.heads = nn.Sequential(*list(model.heads.children())[:-1])
        model = model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        return model

    def process_images(self, image_paths):
        transformations = tr.Compose(
            [
                tr.Resize((518, 518)),
                tr.ToTensor(),
                tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        images = [
            transformations(Image.open(path)).unsqueeze(0) for path in image_paths
        ]
        images = torch.cat(images).to(self.device)
        return images

    def get_embeddings(self, image_paths):
        images = self.process_images(image_paths)
        with torch.no_grad():
            embeddings = self.model(images).detach().cpu()
        return embeddings


def calculate_similarity_between_datasets(dataset_a_dir, dataset_b_dir, title):
    similarity = cosineSimilarity()
    embeddings_a = {}
    embeddings_b = {}

    # Calculate embeddings for dataset A
    for vid_name in os.listdir(dataset_a_dir):
        path = os.path.join(dataset_a_dir, vid_name, "JPEGImages", "0001.jpg")
        if os.path.isfile(path):
            embeddings_a[vid_name] = similarity.get_embeddings([path])

    # Calculate embeddings for dataset B
    for vid_name in os.listdir(dataset_b_dir):
        path = os.path.join(dataset_b_dir, vid_name, "JPEGImages", "0001.jpg")
        if os.path.isfile(path):
            embeddings_b[vid_name] = similarity.get_embeddings([path])

    # Compare between datasets
    similarities = []
    for emb_a in embeddings_a.values():
        for emb_b in embeddings_b.values():
            score = torch.nn.functional.cosine_similarity(emb_a, emb_b).numpy().tolist()
            similarities.append(score)

    similarities = np.array(similarities)
    mean_similarity = np.mean(similarities)

    # Optionally, visualize with histogram
    plt.hist(similarities, bins=20, range=(0, 1))
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Inter-Dataset Cosine Similarity: " + title)
    plt.savefig(f"{title}_histogram.png")
    plt.close()

    return mean_similarity

def calculate_similarity_of_dir(dir_name):
    image_paths = [
        os.path.join(dir_name, vid_name, "JPEGImages", "0001.jpg")
        for vid_name in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, vid_name, "JPEGImages", "0001.jpg"))
    ]
    embeddings = {}
    similarity = cosineSimilarity()

    # Optionally, add tqdm here if processing a large number of images
    # for path in tqdm(image_paths, desc="Processing Images"):
    for path in tqdm(image_paths, desc="Calculating Image Embeddings"):
        embeddings[path] = similarity.get_embeddings([path])

    similarities = []
    # Wrapping the combinations with tqdm for progress visualization
    for (img1, emb1), (img2, emb2) in tqdm(
        itertools.combinations(embeddings.items(), 2), desc="Calculating Similarities"
    ):
        score = torch.nn.functional.cosine_similarity(emb1, emb2).numpy().tolist()
        similarities.append(score)
        
    similarities = np.array(similarities)
    
    if True:
        bins = 20
        if dir_name == "/exports/eddie/scratch/s2062378/dataset/VMD/Pexels":
            bins = 10
        plt.hist(similarities, bins=bins, range=(0, 1))
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        title_map = {
            "Pexels": "Pexels Annotated",
            "test": "VMDD-Test",
            "train": "VMDD-Train",
            "unlabeled": "Pexels Unlabeled"
        }
        plt.title(f"Intra-dataset Cosine Similarity: {title_map[dir_name.split('/')[-1]]}")
        plt.savefig(f"{dir_name}_histogram.png")
        plt.close()

    return np.mean(similarities)


if __name__ == "__main__":
    print("Similarity of VMD TEST:")
    print(calculate_similarity_of_dir("/exports/eddie/scratch/s2062378/dataset/VMD/test"))
    
    print("Similarity of VMD TRAIN:")
    print(calculate_similarity_of_dir("/exports/eddie/scratch/s2062378/dataset/VMD/train"))
    
    print("Similarity of Pexels (unlabeled):")
    print(calculate_similarity_of_dir("/exports/eddie/scratch/s2062378/dataset/VMD/unlabeled"))

    print("Similarity between VMD TEST and TRAIN:")
    print(calculate_similarity_between_datasets("/exports/eddie/scratch/s2062378/dataset/VMD/test", "/exports/eddie/scratch/s2062378/dataset/VMD/train", "VMD Test and VMD Train"))
    
    print("Similarity between VMD TEST and UNLABELED:")
    print(calculate_similarity_between_datasets("/exports/eddie/scratch/s2062378/dataset/VMD/test", "/exports/eddie/scratch/s2062378/dataset/VMD/unlabeled", "VMD Test and Unlabeled"))
    
    print("Similarity between VMD TRAIN and UNLABELED:")
    print(calculate_similarity_between_datasets("/exports/eddie/scratch/s2062378/dataset/VMD/train", "/exports/eddie/scratch/s2062378/dataset/VMD/unlabeled", "VMD Train and Unlabeled"))