import gc
from typing import List

import PIL.Image as Image
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from tqdm import tqdm


class DINOv2Args:
    dtype: torch.dtype = torch.float32

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the DINO model parameters."""
        return {
            "dtype": cls.dtype,
        }



@torch.no_grad()
def extract_dinov2_features(image_paths: List[str], device: torch.device) -> torch.Tensor:
    dinov2_feat_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    dinov2_feat_extractor.to(dtype=DINOv2Args.dtype)

    imgs = np.stack([cv2.imread(str(img_path)) for img_path in image_paths])
    K, H, W, _ = imgs.shape
    
    patch_h = H // 10
    patch_w = W // 10
    # feat_dim = 384 # vits14
    # feat_dim = 768 # vitb14
    feat_dim = 1024 # vitl14
    # feat_dim = 1536 # vitg14
    
    transform = T.Compose([
        # T.GaussianBlur(9, sigma=(0.1, 2.0)),
        T.Resize((patch_h * 14, patch_w * 14)),
        T.CenterCrop((patch_h * 14, patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    imgs_tensor = torch.zeros((K, 3, patch_h * 14, patch_w * 14), device=device)
    for j in range(K):
        img = Image.fromarray(imgs[j])
        imgs_tensor[j] = transform(img)[:3]
    with torch.no_grad():
        features_dict = dinov2_feat_extractor.forward_features(imgs_tensor.to(dtype=DINOv2Args.dtype))
        features = features_dict['x_norm_patchtokens']
        features = features.reshape((K, patch_h, patch_w, feat_dim))
    
    del dinov2_feat_extractor
    del imgs_tensor
    return features
