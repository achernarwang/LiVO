import os
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore


class MethodScoringDataset(Dataset):
    def __init__(self, imgs_dirs: list[str], method_name: str) -> None:
        self.pil2tensor = v2.PILToTensor()
        self.imgs_dirs = imgs_dirs
        self.all_imgs_path = []
        self.all_prompts = []
        
        for imgs_dir in self.imgs_dirs:
            imgs_dir = Path(imgs_dir) 
            with open(imgs_dir / 'prompt.jsonl', 'r') as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                prompt_data = json.loads(line)
                self.all_prompts.extend([prompt_data['prompt'] for _ in range(50)]) 
                self.all_imgs_path.extend([imgs_dir / f"{idx}_{i}.jpg" for i in range(50)])
    
    def __len__(self):
        return len(self.all_imgs_path)

    def __getitem__(self, index):
        return self.pil2tensor(Image.open(self.all_imgs_path[index]).convert("RGB")), self.all_prompts[index]

class InceptionScoring:
    def __init__(self, device='cuda:0') -> None:
        self.inception = InceptionScore().to(device=device)
        self.device = device
    
    def compute(self, imgs_dirs: list[str], method_name: str, batch_size=64, num_workers=4):
        dataset = MethodScoringDataset(imgs_dirs, method_name)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        
        for image_batch, _ in tqdm(data_loader):
            self.inception.update(image_batch.to(self.device))

        isc_mean, isc_std = self.inception.compute()
        self.inception.reset()
        return float(isc_mean), float(isc_std)

class CLIPScoring:
    def __init__(self, device='cuda:0'):
        self.clip = CLIPScore("openai/clip-vit-large-patch14").to(device=device)
        self.device = device
    
    def compute(self, imgs_dirs: list[str], method_name: str, batch_size=64, num_workers=4):
        dataset = MethodScoringDataset(imgs_dirs, method_name)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        
        for images, prompts in tqdm(data_loader):
            self.clip.update(images.to(self.device), list(prompts))
        
        scores = self.clip.compute()
        self.clip.reset()
        return float(scores)

class FIDScoring:
    def __init__(self, device='cuda:0') -> None:
        self.fid = FrechetInceptionDistance().to(device=device)
        self.device = device
        

    def compute(self, imgs_dirs: list[str], ref_imgs_dirs: list[str], method_name: str, ref_method_name: str, batch_size=64, num_workers=4):
        dataset_1 = MethodScoringDataset(imgs_dirs, method_name)
        dataset_2 = MethodScoringDataset(ref_imgs_dirs, ref_method_name)
        data_loader_1 = DataLoader(dataset_1, batch_size=batch_size, num_workers=num_workers)
        data_loader_2 = DataLoader(dataset_2, batch_size=batch_size, num_workers=num_workers)

        for image_batch, _ in tqdm(data_loader_1):
            self.fid.update(image_batch.to(self.device), real=False)
        
        for image_batch, _ in tqdm(data_loader_2):
            self.fid.update(image_batch.to(self.device), real=True)
        
        scores = self.fid.compute()
        self.fid.reset()
        return float(scores)
    
def argparsing():
    parser = argparse.ArgumentParser(description="Evaluating image quality metrics")
    parser.add_argument("--metrics", type=str, nargs="+", required=True, choices=["isc", "fid", "clip"], help="Metrics to evaluate")
    parser.add_argument("--method", type=str, required=True, choices=["sd-1-5", "livo"], help="Method to evaluate")
    parser.add_argument("--ref_method", type=str, default="sd-1-5", choices=["sd-1-5"], help="Reference method for FID evaluation")
    parser.add_argument("--device", type=str, required=True, default='cuda:0', help="device to run the evaluation")
    parser.add_argument("--eval_image_paths", type=str, nargs="+", required=True, help="List of evaluation image paths")
    parser.add_argument("--ref_image_paths", type=str, nargs="+", help="List of reference image paths")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")

    args = parser.parse_args()

    for path in args.eval_image_paths:
        if not os.path.exists(path) or not os.path.isdir(path):
            parser.error(f"Path {path} does not exist.")
    
    if 'fid' in args.metrics:
        if args.ref_image_paths is None:
            parser.error("Reference image paths are required for FID evaluation")
        else:
            for path in args.ref_image_paths:
                if not os.path.exists(path) or not os.path.isdir(path):
                    parser.error(f"Path {path} does not exist.")

    return args

def main():
    args = argparsing()

    if 'isc' in args.metrics:
        print("Computing Inception Score")
        inception_scoring = InceptionScoring(device=args.device)
        isc_mean, isc_std = inception_scoring.compute(args.eval_image_paths, args.method, args.batch_size, args.num_workers)
        print(f"{args.method} isc_mean: {isc_mean}, isc_std: {isc_std}")
        del inception_scoring
        torch.cuda.empty_cache()

    if 'fid' in args.metrics:
        print("Computing FID Score")
        fid_scoring = FIDScoring(device=args.device)
        fid_score = fid_scoring.compute(args.eval_image_paths, args.ref_image_paths, args.method, args.ref_method, args.batch_size, args.num_workers)
        print(f"{args.method} vs {args.ref_method} fid_score: {fid_score}")
        del fid_scoring
        torch.cuda.empty_cache()

    if 'clip' in args.metrics:
        print("Computing CLIP Score")
        clip_scoring = CLIPScoring(device=args.device)
        clip_score = clip_scoring.compute(args.eval_image_paths, args.method, args.batch_size, args.num_workers)
        print(f"{args.method} clip_score: {clip_score}")
        del clip_scoring
        torch.cuda.empty_cache()

if __name__ == '__main__':    
    main()