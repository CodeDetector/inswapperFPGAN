# from gfpgan import GFPGANer
import os 
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2 
from .utils import GFPGANer 

def bg_sampler(input_img  , bg_upsampler = None  ): 
    
    bg_tile = 400 
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    
    
   
    
    # GFPGan version default = 1.3 
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
    # url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    
    model_path = os.path.join('./gfp/weights/' +  model_name + '.pth')
    
    if bg_upsampler  : 
        bg_upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        # bg_tile default = 400 
        tile=bg_tile,
        tile_pad=10,
        pre_pad=0,
        half=False)
        restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch=arch,
            channel_multiplier=channel_multiplier , 
            bg_upsampler= bg_upsampler)
    else : 
        restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch=arch,
            channel_multiplier=channel_multiplier )
        
        # restore faces and background if necessary
        
    _, _, restored_img = restorer.enhance(
        input_img,
        has_aligned=False ,
        only_center_face=False ,
        paste_back=True,
        weight=0.5 
        )
     
    return restored_img 

