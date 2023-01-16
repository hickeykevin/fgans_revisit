import torch
import torch.nn as nn

import numpy as np
import scipy.stats
import tarfile

from pathlib import Path, PurePath
from tqdm import trange
from typing import Dict, Tuple
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import convert_image_dtype, to_pil_image, pil_to_tensor

def prepare_for_torchmetric_FID(imgs: torch.Tensor, make_between_01: bool = True):
    '''
    used for in training procedure of calculating fid 

    args:
        imgs: batch of generated images to be resized
        make_between_01: True if generated images in range [-1, 1]
    returns:
        results: torch.tensor of imgs of size 299x299 in range [0, 255]
    '''
    if make_between_01:
        imgs = (imgs + 1) / 2
    print("INFO: Preparing imgs to be used in torchmetrics FID evaluation")
    for idx, img in enumerate(tqdm(imgs)):
        img = to_pil_image(img).resize((299, 299))
        img = convert_image_dtype(pil_to_tensor(img), dtype=torch.uint8)
        imgs[idx] = img

    return imgs

    


def unpickle(file) -> Dict: #return dict as {"data": np.array, "labels": list}
    with open(file, 'rb') as fo:
        result = pickle.load(fo, encoding='bytes')
    return result

def unpack_cifar10_tar_file(data_folder_path="./data/cifar10"):
    data_folder_path = Path(data_folder_path)
    dataset_name = "cifar10"
    extension = "cifar-10-"
    file = tarfile.open(data_folder_path / f"{extension}python.tar.gz")
    file.extractall(data_folder_path / "tmp")
    file.close()

    print(f"[INFO] Saving all images in {dataset_name} to disc")
    save_path = data_folder_path / "raw"
    save_path.mkdir(parents=True, exist_ok=True)
    
    num = 0
    for batch in [f"data_batch_{x}" for x in range(1, 6)]:
        items = unpickle(str(data_folder_path / "tmp" /f"{extension}batches-py/{batch}"))
        imgs = items[b'data']
        for img in tqdm(imgs):
            pxls_R = img[0:1024].reshape((32, 32))
            pxls_G = img[1024:2048].reshape((32, 32))
            pxls_B = img[2048:3072].reshape((32, 32))
            img = np.dstack((pxls_R, pxls_G, pxls_B))
            pil_img = Image.fromarray(img.astype('uint8'), mode="RGB")
            pil_img.save(save_path / f"{num}.png")
            num += 1


def resize_raw_images(data_folder_path: Path, resized_img_size: int):
    raw_data_path = data_folder_path / "raw"
    resized_data_folder_path = data_folder_path / f"resized_{resized_img_size}"
    resized_data_folder_path.mkdir(exists_ok=True, parents=True)
    
    all_original_imgs = raw_data_path.glob("*.png")
    for i, img in enumerate(tqdm(all_original_imgs)):
        pil_img = Image.open(str(img)).resize((resized_img_size, resized_img_size))
        pil_img.save(resized_data_folder_path / f"{i}.png")

    assert len(all_original_imgs) == len(resized_data_folder_path.glob("*.png"))
    return resized_data_folder_path


def prepare_cleanfid_fake_directory(params: Dict):
    fake_folder_location = Path.cwd() / "data" / f"{params['dataset']}" / "fake_imgs"
    if not fake_folder_location.exists():
        fake_folder_location.mkdir(exist_ok=True, parents=True)
    return fake_folder_location


def save_generated_img_to_folder(img: torch.Tensor, idx: int, img_size: int, save_path):
    '''
    function to save generated images to disc; used for cleanfid calulation

    args:
        img: single generated img as torch tensor
        idx: the i'th image in the dataset; used for naming 
        params: overall configuration dictionary
    '''
    
    img = to_pil_image(img).resize((img_size, img_size))
    img.save(str(save_path / f"{idx}.png"))



def generate_imgs(net_G, device, params: Dict, size=50000, batch_size=128):
    '''
    generate fake instances from Generator network;
    can be used in training process or with saved network weights after training

    args:
        net_G: Generator network pytorch class (dcgan)
        device: gpu/cpu device
        params: Overall configuration as dictionary
        size: number of generated samples to create
        batch_size: number of samples to create for each iteration

    returns:
        imgs: torch tensor of all generated images; all dimensionality and value ranges stay same
    '''
    net_G.eval()
    imgs_list = []
    with torch.no_grad():
        for start in trange(0, size, batch_size,
                            desc='Evaluating', ncols=0, leave=False):
            end = min(start + batch_size, size)
            z = torch.randn((end - start, params['z_dim'])).to(device) 
            imgs = net_G(z).cpu()

            if params['fid_method'] == "cleanfid":
                for idx, img in enumerate(imgs):
                    idx = start + idx
                    save_generated_img_to_folder(img, idx=idx, params=params)
            else:
                imgs_list.append(imgs)
                result = torch.stack(imgs, dim=0)
                return result


    #imgs = (imgs + 1) / 2  # is this needed for Jan_JSD_DCGAN_FID experiment? 
    # No, i think it's for a revised architecture, PGC repo architectures
    #net_G.train()
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1 or classname.find('Linear')!=-1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)
        
        
def calc_base_is_fid(Q_net, div, data_set, base_directory, vmf, run_list, device, stats_name, params):
    Q_net.to(device)
    metrics_dict = {"IS": [], "IS_std": [], "FID": []}
    SAVED_MODELS_PATH = Path.cwd() / "logs" / data_set / base_directory
    FID_STATS = Path.cwd() / "fid_stats"

    for run in run_list:
        
        model_name = f'Q_f-GAN-{div}-DCGAN_{run}_.pth'
        pure_path = PurePath(SAVED_MODELS_PATH, model_name)
        Q_net.load_state_dict(torch.load(str(pure_path))) #confirm this path is correct

        Q_net.train()
        with torch.no_grad():
            if vmf:
                p_z = HypersphericalUniform(params['nz'] - 1, device=device)#VonMisesFisher(d_mean, torch.zeros(params['bsize'], 1))
                dec_noise = p_z.sample(10000)#p_z.sample()
                noise_z = dec_noise.view(dec_noise.shape[0], dec_noise.shape[1], 1, 1)
            else: 
                noise_z = torch.randn(10000, params['nz'], 1, 1, device=device) 
            fake_imgs = Q_net(noise_z).detach()

        z_fake_imgs = fake_imgs
	
        (IS, IS_std), FID = get_inception_score_and_fid(z_fake_imgs, str(FID_STATS / stats_name)) # adjust this path
        metrics_dict["IS"].append(IS)
        metrics_dict["IS_std"].append(IS_std)
        metrics_dict["FID"].append(FID)
        torch.cuda.empty_cache()
        
    return metrics_dict

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, h