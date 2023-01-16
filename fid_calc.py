
import torch
from models import  Q_DCGAN
from pathlib import Path, PurePath
from pytorch_gan_metrics import get_fid
from pytorch_gan_metrics.utils import calc_and_save_stats
from models import Q_DCGAN
import numpy as np
from config import Config
from collections import namedtuple
from dataclasses import asdict
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Dict
import cleanfid
from utils import unpack_cifar10_tar_file, resize_raw_images, save_cifar10_64x64_imgs, prepare_cleanfid_fake_directory, save_generated_img_to_folder
from torchmetrics.image.fid import FrechetInceptionDistance
from dataloaders import CIFAR10DataModule


# think about how to fix this
def torchmetrics_caclulate_FID(dataset_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    params = asdict(Config())
    if dataset_name == "cifar10":
        train_set, train_loader, *_ = CIFAR10DataModule(params=params)
        pass

def clean_fid_precompute_stats(params: Dict, img_size: int=32):
    '''
    Precompute cifar10 statistics for specified image size.
    Stats calculated from specified images. 
    '''
    from cleanfid import fid
    data_folder_path = Path.cwd() / "data" / params['dataset']
    print("[INFO]: Unpacking tar file to save raw images")
    unpack_cifar10_tar_file(str(data_folder_path))

    print(f"[INFO]: Saving raw imgs to {img_size}x{img_size} sized imgs")
    resized_img_folder_path = resize_raw_images(data_folder_path=data_folder_path, resized_img_size=img_size)

    # cleanfid stats name saved to cleanfid.__file__/stats
    print("[INFO]: Calculating real dataset statistics on newly resized dataset")
    fid.make_custom_stats(
        params['eval_stats_name'],
        str(resized_img_folder_path),
        mode='clean'
        )



def clean_fid_compute_fid(generated_imgs: torch.tensor, params, img_size: int=32):
    from cleanfid import fid

    # cleanfid stats name located at cleanfid.__file__/stats
    cleanfid_stats_location = Path(cleanfid.__file__) / "stats"
    if not Path(cleanfid_stats_location / f"{params['eval_stats_name']}").exists():
        clean_fid_precompute_stats(img_size=img_size)
    
    fake_imgs_path = prepare_cleanfid_fake_directory()
    for idx, img in generated_imgs:
        save_generated_img_to_folder(img, idx, img_size, fake_imgs_path)

    score = fid.compute_fid(
        str(fake_imgs_path),
        dataset_name=params['eval_stats_name'],
        mode='clean',
        dataset_split='custom'
        )
    return score
    

        
def pgm_precompute_statstistics(params: Dict, img_size: int):
    '''
    Precompute cifar10 statistics for specified image size. 
    '''
    original_sized_imgs = Path.cwd() / "data" / "raw"
    output_path = Path.cwd() / "fid_stats" / params['eval_stats_name']
    calc_and_save_stats(
        input_path=original_sized_imgs,
        output_path=output_path,
        img_size=img_size,
        batch_size=50,
        use_torch=True
        )
    return output_path


def pgm_calc_fid(imgs: torch.Tensor, make_between_01: bool, params: Dict):
    if make_between_01:
        imgs = (imgs + 1) / 2
    assert 0 <= imgs.min() and imgs.max() <= 1
    stats_name = Path.cwd() / "fid_stats" / f"{params['eval_stats_name']}.npz"
    FID = get_fid(
        imgs, 
        str(stats_name))
    return FID


def calc_fid_inception(Q_net, Q_models: list, device, n_samples: int, config):
    '''
    Args:
        Q_net: Specific experiment's Generator class architecture. Ex Q_DCGAN, PGMGenerator
        Q_models: list of saved Q_net checkpoints to load and evaluate
        device: cuda device
        n_samples: number of samples to generate and calculate FID/Inception Scores with. 
        config: Configuration dataclass imported from config.py
        
    '''
    Q_net.to(device)
    fid_stats = Path.cwd() / "fid_stats" / config.eval_stats_name
    
    Result = namedtuple("Result", field_names=["IS_mean", "IS_std", "FID"])
    results = []
    for model in tqdm(Q_models):
        Q_net.load_state_dict(torch.load(str(model)))
        Q_net.eval()
        with torch.no_grad():
            if config.use_vMF:
                p_z = HypersphericalUniform(config.nz - 1, device=device)#VonMisesFisher(d_mean, torch.zeros(config.bsize, 1))
                dec_noise = p_z.sample(n_samples)#p_z.sample()
                noise_z = dec_noise.view(dec_noise.shape[0], dec_noise.shape[1], 1, 1)
            else: 
                noise_z = torch.randn(n_samples, config.nz, 1, 1, device=device) 
            fake_imgs = Q_net(noise_z)
            print(f"Img sizes: {fake_imgs.size()}")

        (IS, IS_std), FID = get_inception_score_and_fid(fake_imgs, str(fid_stats)) # adjust this path
        Res = Result(IS, IS_std, FID)
        results.append(Res)
        print(Res)
        torch.cuda.empty_cache()
        
    return results


if __name__ == "__main__": 
    custom_stats_calculation("cifar10")
    
    # conf = Config()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # params = asdict(conf)
    
    # print("[INFO]: Begin eval metrics calculations")
    # print(f"{conf.dataset}, {conf.exp_name}")

    # models_dir = Path.cwd() / "logs" / conf.dataset / conf.exp_name
    # saved_checkpoints = [str(x) for x in models_dir.glob("*.pth")]
    # Q_models = [x for x in saved_checkpoints if "Q" in x.split()[-1]][90:100][::2]


    # results = calc_fid_inception(
    #     Q_net=Q_DCGAN(params=params),
    #     Q_models=Q_models,
    #     n_samples=20000,
    #     device=device,
    #     config=conf
    # )
    # print("[INFO]: Complete")    




