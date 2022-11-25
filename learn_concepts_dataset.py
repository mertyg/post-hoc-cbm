import os
import pickle
import torch
import argparse

import numpy as np

from models import get_model
from concepts import learn_concept_bank
from data import get_concept_loaders


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--dataset-name", default="cub", type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--num-workers", default=4, type=int, help="Number of workers in the data loader.")
    parser.add_argument("--batch-size", default=100, type=int, help="Batch size in the concept loader.")
    parser.add_argument("--C", nargs="+", default=[0.01, 0.1], type=float, help="Regularization parameter for SVMs.")
    parser.add_argument("--n-samples", default=50, type=int, 
                        help="Number of positive/negative samples used to learn concepts.")
    return parser.parse_args()

def main():
    args = config()
    n_samples = args.n_samples

    # Bottleneck part of model
    backbone, preprocess = get_model(args, args.backbone_name)
    backbone = backbone.to(args.device)
    backbone = backbone.eval()
    
    concept_libs = {C: {} for C in args.C}
    # Get the positive and negative loaders for each concept. 
    
    concept_loaders = get_concept_loaders(args.dataset_name, preprocess, n_samples=args.n_samples, batch_size=args.batch_size, 
                                          num_workers=args.num_workers, seed=args.seed)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for concept_name, loaders in concept_loaders.items():
        pos_loader, neg_loader = loaders['pos'], loaders['neg']
        # Get CAV for each concept using positive/negative image split
        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, n_samples, args.C, device="cuda")
        
        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        for C in args.C:
            concept_libs[C][concept_name] = cav_info[C]
            print(concept_name, C, cav_info[C][1], cav_info[C][2])

    # Save CAV results    
    for C in concept_libs.keys():
        lib_path = os.path.join(args.out_dir, f"{args.dataset_name}_{args.backbone_name}_{C}_{args.n_samples}.pkl")
        with open(lib_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved to: {lib_path}")        
    
        total_concepts = len(concept_libs[C].keys())
        print(f"File: {lib_path}, Total: {total_concepts}")

if __name__ == "__main__":
    main()
