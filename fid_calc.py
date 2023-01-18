
import torch
from torch.nn import DataParallel
from models import  Q_DCGAN
from pathlib import Path
from models import Q_DCGAN
from config import Config
from dataclasses import asdict
from tqdm import tqdm
from pathlib import Path
from utils import prepare_for_torchmetric_FID, generate_imgs
import re

def main(experiment):
    params = asdict(Config())
    weights_path = Path(experiment)
    generator_weights = [str(x) for x in weights_path.iterdir() if x.name.startswith("Q_")]

    # Ensure the weights are ordered by epoch number
    atoi = lambda text: int(text) if text.isdigit() else text
    nautral_keys = lambda text: [atoi(c) for c in re.split(r'(\d+)', text)]
    generator_weights.sort(key=nautral_keys)

    # set model and FID to device
    device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load in precomputed statstics for particular dataset
    print(f"[INFO] Loading in precomputed statistics for {params['dataset']} dataset")
    stats_file = Path.cwd() / "fid_stats" / params['eval_stats_name']
    FID = torch.load(stats_file)
    FID.to(device=device_0)
    Q_net = Q_DCGAN(params).to(device=device_0)

    print("[INFO] Calculating FID on saved model weights")
    for idx, weight in enumerate(generator_weights[::params['fid_eval_n_epochs']]):
        idx = idx*params['fid_eval_n_epochs']
        Q_net.load_state_dict(torch.load(str(weight)))
        fid_fake_loader = generate_imgs(Q_net, device=device_0, params=params, size=50000)
        for batch in tqdm(fid_fake_loader, desc=f"Epoch={idx}"):
            fid_fake = prepare_for_torchmetric_FID(batch[0], make_between_01=True).to(device=device_0)
            FID.update(fid_fake, real=False)
    
        torchmetrics_fid_score = FID.compute()
        print(f"Epoch={idx} FID Score: {torchmetrics_fid_score.item()}")
        FID.reset()
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help="path to experiment and generator weights")
    args = parser.parse_args()
    main(experiment=args.exp)
        

    





