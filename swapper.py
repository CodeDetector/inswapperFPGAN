"""
This project is developed by Haofan Wang to support face swap in single frame. Multi-frame will be supported soon!

It is highly built on the top of insightface, sd-webui-roop and CodeFormer.
"""

import os
import cv2
import copy
import argparse
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple
import matplotlib.pyplot as plt 
from gfp import bg_sampler 
import torch 
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2 
from gfp.utils import GFPGANer 

def compare_faces(face , reference_face , face_distance : float) -> bool:
	current_face_distance = calc_face_distance(face, reference_face)
	return current_face_distance < face_distance


def calc_face_distance(face , reference_face ) -> float:
	if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
		return 1 - np.dot(face.normed_embedding, reference_face.normed_embedding)
	return 0



def getFaceSwapModel(model_path: str):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = insightface.model_zoo.get_model(model_path)
    return model


def getFaceAnalyser(model_path: str, providers,
                    det_size=(320, 320)):
    print("The provider is : " , providers)
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser,
                 frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

    
def get_many_faces(face_analyser,
                   frame:np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper,
              source_face,
              target_face,
              temp_frame):
    """
    paste source_face on target image
    """
    # print("The sourcefaces are -- -- - -- " ,source_faces) 
    # print("the targetfaces are ----- - -- -- -- - -" , target_faces )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        output = face_swapper.get(temp_frame, source_face, target_face,  paste_back=True )
        return output 
    return None

def is_video(path) : 
    return True  

def get_video_frame(video_path : str, frame_number : int = 0):
    frames = []
    
    if is_video(video_path):
      video_capture = cv2.VideoCapture(video_path) 
      video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        # Read all the frames from the video
        
      while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    if len(frames) != 0 : 
        return frames   
    return None

def create_video(new_frames ) : 
    
    h, w= new_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    fps = 25
    # size = (image_list[0].width, image_list[0].height)
    out = cv2.VideoWriter('output1.mp4', fourcc, fps , (w, h))

    # Write each image to the video
    for img in new_frames:
        out.write(img)

    # Release the video writer
    out.release()

    
def process_video(video_path , target_img_path , reference_img_path, model , restore , bg_upsampler  ) : 
    
   
    target_imgs = [Image.open(img_path) for img_path in target_img_path]
    reference_imgs = [Image.open(img_path) for img_path in reference_img_path]
    
    # print("These are the target images " , target_imgs[0] )
    providers = onnxruntime.get_available_providers()

    # load face_analyser
    face_analyser = getFaceAnalyser(model, providers)
    
    # load face_swapper
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    
    face_swapper = getFaceSwapModel(model_path)
    
    frames = get_video_frame(video_path)
    
    
    
    target_faces = [get_one_face(face_analyser= face_analyser  , frame=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)) for img in target_imgs]  
    reference_faces = [get_one_face(face_analyser= face_analyser  , frame=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)) for img in reference_imgs ] 
    
    bg_tile = 400 
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    # GFPGan version default = 1.3 
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
    # url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    
    model_path = os.path.join('./gfp/weights/' +  model_name + '.pth')
    restorer = None 
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
        
    if isinstance(frames , list) and len(frames) : 
        for idx, frame in enumerate(frames) : 
            print(f"Processing image {idx}") 
            img = process_image(frame , target_faces , reference_faces=reference_faces , face_analyser= face_analyser , face_swapper= face_swapper , restore = restore )
            print(f"Restoring image {idx}")
            img = bg_sampler(img , restorer= restorer) 
            print(f"Saving image {idx}") 
            cv2.imwrite("./images"+str(idx)+".png",img)
            del img
    else : 
        return 0 
    
    # create_video(new_frames)
    
    return 1 




def process_image(frame : Image.Image,
            target_faces: Union[Image.Image , List],
            reference_faces : Union[Image.Image , List] ,
            face_analyser , face_swapper , restore : bool):
    
    # read target image
    frame  = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    # print(frame.shape  )
    # detect faces that will be replaced in the target image
    frame_faces = get_many_faces(face_analyser, frame)
    
    if frame_faces is not None:
        # temp_frame = copy.deepcopy(frame)
        if isinstance(frame_faces, list):
            for face in frame_faces :
                for i in range(len(target_faces))  : 
                    t_face = target_faces[i]
                    if compare_faces(face , t_face , 0.7 ) : 
                        frame = swap_face(
                            face_swapper,
                            face ,
                            reference_faces[i],
                            frame
                        )
                        break 
    print("face ....  .. .. ")
    
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

