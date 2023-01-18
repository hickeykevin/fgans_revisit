from torchmetrics.image.fid import FrechetInceptionDistance
from dataloaders import CIFAR10DataModule
import torch
from pathlib import Path
import argparse
from config import Config
from dataclasses import asdict
from utils import prepare_for_torchmetric_FID
from tqdm import tqdm
import pickle
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("stats_name")
# args = parser.parse_args()

def main():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = Config()
    params = asdict(config)
    
    data_path = Path.cwd() / "data" / params['dataset']
    if params['dataset'] == "cifar10":
        train_set, train_loader, *_ = CIFAR10DataModule(params, PATH=data_path)
    
    filename = Path.cwd() / "fid_stats" / params['eval_stats_name']
    if filename.exists():
        print("File already exists; do you want to overwrite the file? If so, delete existing file and run script again")
        raise FileExistsError
        
    FID = FrechetInceptionDistance(reset_real_features=False, normalize=False).to(device)
    
    print(f"[INFO] Computing real data statistics for {params['dataset']} dataset")
    for batch in tqdm(train_loader):
        data = batch[0]
        data = prepare_for_torchmetric_FID(data,  make_between_01=True).to(device)
        FID.update(data, real=True)
    
    print(f"[INFO] Saving FID statistics module to {str(filename)}")     
    torch.save(FID, filename)
    print("[INFO] Complete")

if __name__ == "__main__":
    main()
