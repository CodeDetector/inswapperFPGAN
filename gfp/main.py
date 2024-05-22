# from gfpgan import GFPGANer
import os 
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2 
from .utils import GFPGANer 
import torch 

def bg_sampler(input_img  ,restorer ,  bg_upsampler = None  ):         
        # restore faces and background if necessary
    with torch.inference_mode():
        _, _, restored_img = restorer.enhance(
            input_img,
            has_aligned=False ,
            only_center_face=False ,
            paste_back=True,
            weight=0.5 
            )
     
        return restored_img 

