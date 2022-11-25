import os
import pickle
import torch

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from .constants import CUB_PROCESSED_DIR


def cub_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed):
    from .cub import CUBConceptDataset, get_concept_dicts
    TRAIN_PKL = os.path.join(CUB_PROCESSED_DIR, "train.pkl")
    metadata = pickle.load(open(TRAIN_PKL, "rb"))

    concept_info = get_concept_dicts(metadata=metadata)

    np.random.seed(seed)
    torch.manual_seed(seed)
    concept_loaders = {}
    for c_idx, c_data in concept_info.items():
        pos_ims, neg_ims = c_data[1], c_data[0]
        # Sample equal number of positive and negative images
        try:
            pos_concept_ims = np.random.choice(pos_ims, 2*n_samples, replace=False)
            neg_concept_ims = np.random.choice(neg_ims, 2*n_samples, replace=False)
        except Exception as e:
            print(e)
            print(f"{len(pos_ims)} positives, {len(neg_ims)} negatives")
            pos_concept_ims = np.random.choice(pos_ims, 2*n_samples, replace=True)
            neg_concept_ims = np.random.choice(neg_ims, 2*n_samples, replace=True)

        pos_ds = CUBConceptDataset(pos_concept_ims, preprocess)
        neg_ds = CUBConceptDataset(neg_concept_ims, preprocess)
        pos_loader = DataLoader(pos_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        neg_loader = DataLoader(neg_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        concept_loaders[c_idx] = {
            "pos": pos_loader,
            "neg": neg_loader
        }
    return concept_loaders  
    
    
def derm7pt_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed):
    from .derma_data import Derm7ptDataset
    from .constants import DERM7_META, DERM7_TRAIN_IDX, DERM7_VAL_IDX, DERM7_FOLDER
    df = pd.read_csv(DERM7_META)
    train_indexes = list(pd.read_csv(DERM7_TRAIN_IDX)['indexes'])
    val_indexes = list(pd.read_csv(DERM7_VAL_IDX)['indexes'])
    print(df.columns)
    df["TypicalPigmentNetwork"] = df.apply(lambda row: {"absent": 0, "typical": 1, "atypical": -1}[row["pigment_network"]] ,axis=1)
    df["AtypicalPigmentNetwork"] = df.apply(lambda row: {"absent": 0, "typical": -1, "atypical": 1}[row["pigment_network"]] ,axis=1)

    df["RegularStreaks"] = df.apply(lambda row: {"absent": 0, "regular": 1, "irregular": -1}[row["streaks"]] ,axis=1)
    df["IrregularStreaks"] = df.apply(lambda row: {"absent": 0, "regular": -1, "irregular": 1}[row["streaks"]] ,axis=1)

    df["RegressionStructures"] = df.apply(lambda row: (1-int(row["regression_structures"] == "absent")) ,axis=1)

    df["RegularDG"] = df.apply(lambda row: {"absent": 0, "regular": 1, "irregular": -1}[row["dots_and_globules"]] ,axis=1)
    df["IrregularDG"] = df.apply(lambda row: {"absent": 0, "regular": -1, "irregular": 1}[row["dots_and_globules"]] ,axis=1)

    df["BWV"] = df.apply(lambda row: {"absent": 0, "present": 1}[row["blue_whitish_veil"]] ,axis=1)

    df = df.iloc[train_indexes+val_indexes]

    concepts = ["BWV", "RegularDG", "IrregularDG", "RegressionStructures", "IrregularStreaks",
                   "RegularStreaks", "AtypicalPigmentNetwork", "TypicalPigmentNetwork"]
    concept_loaders = {}
    
    for c_name in concepts: 
        pos_df = df[df[c_name] == 1]
        neg_df = df[df[c_name] == 0]
        base_dir = os.path.join(DERM7_FOLDER, "images")
        image_key = "derm"

        print(pos_df.shape, neg_df.shape)
        
        if (pos_df.shape[0] < 2*n_samples) or (neg_df.shape[0] < 2*n_samples):
            print("\t Not enough samples! Sampling with replacement")
            pos_df = pos_df.sample(2*n_samples, replace=True)
            neg_df = neg_df.sample(2*n_samples, replace=True)
        else:
            pos_df = pos_df.sample(2*n_samples)
            neg_df = neg_df.sample(2*n_samples)
        
        pos_ds = Derm7ptDataset(pos_df, base_dir=base_dir, image_key=image_key, transform=preprocess)
        neg_ds = Derm7ptDataset(neg_df, base_dir=base_dir, image_key=image_key, transform=preprocess)
        pos_loader = DataLoader(pos_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

        neg_loader = DataLoader(neg_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
        concept_loaders[c_name] = {
            "pos": pos_loader,
            "neg": neg_loader
        }
    return concept_loaders
    
class ListDataset:
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        return image

def broden_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed):
    from .constants import BRODEN_CONCEPTS
    concept_loaders = {}
    concepts = [c for c in os.listdir(BRODEN_CONCEPTS) if os.path.isdir(os.path.join(BRODEN_CONCEPTS, c))]
    for concept_name in concepts:
        pos_dir = os.path.join(BRODEN_CONCEPTS, concept_name, "positives")
        pos_images = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)]
        if (len(pos_images) < 2*n_samples):
            print(f"\t Not enough positive samples for {concept_name}: {len(pos_images)}! Sampling with replacement")
            pos_images = np.random.choice(pos_images, 2*n_samples, replace=True)
        else:
            pos_images = np.random.choice(pos_images, 2*n_samples, replace=False)
        neg_dir = os.path.join(BRODEN_CONCEPTS, concept_name, "negatives")
        neg_images = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)]
        if (len(neg_images) < 2*n_samples):
            print(f"\t Not enough negative samples for {concept_name}: {len(neg_images)}! Sampling with replacement")
            neg_images = np.random.choice(neg_images, 2*n_samples, replace=True)
        else:
            neg_images = np.random.choice(neg_images, 2*n_samples, replace=False)

        pos_ds = ListDataset(pos_images, transform=preprocess)
        neg_ds = ListDataset(neg_images, transform=preprocess)
        pos_loader = DataLoader(pos_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

        neg_loader = DataLoader(neg_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
        concept_loaders[concept_name] = {
            "pos": pos_loader,
            "neg": neg_loader
        }
    return concept_loaders
        
    
def get_concept_loaders(dataset_name, preprocess, n_samples=50, batch_size=100, num_workers=4, seed=1):
    if dataset_name == "cub":
       return cub_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed)
    
    elif dataset_name == "derm7pt":
        return derm7pt_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed)
    
    elif dataset_name == "broden":
        return broden_concept_loaders(preprocess, n_samples, batch_size, num_workers, seed)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    