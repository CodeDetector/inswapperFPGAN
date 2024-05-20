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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return face_swapper.get(temp_frame, source_face, target_face,  paste_back=True)

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
    fps = 30
    # size = (image_list[0].width, image_list[0].height)
    out = cv2.VideoWriter('output1.mp4', fourcc, fps , (w, h))

    # Write each image to the video
    for img in new_frames:
        out.write(img)

    # Release the video writer
    out.release()

    
def process_video(video_path , target_img_path , reference_img_path, model , restore  ) : 
    
    # target_img_paths = args.source_img.split(';')
    # print("Source image paths:", target_img_paths)
    
    # reference_img_paths = args.reference_img.split(';') 
    # print("Reference image paths" , reference_img_paths)
    # print(target_img_path , reference_img_path )
   
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
    
    new_frames = [] 
    
    target_faces = [get_one_face(face_analyser= face_analyser  , frame=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)) for img in target_imgs]  
    reference_faces = [get_one_face(face_analyser= face_analyser  , frame=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)) for img in reference_imgs ] 
    
    if isinstance(frames , list) and len(frames) : 
        for frame in frames : 
            new_frames.append( process_image(frame , target_faces , reference_faces=reference_faces , face_analyser= face_analyser , face_swapper= face_swapper , restore = restore ))
    else : 
        return None 
    
    # create_video(new_frames)
    
    return new_frames 




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
        temp_frame = copy.deepcopy(frame)
        if isinstance(frame_faces, list):
            for face in frame_faces :
                for i in range(len(target_faces))  : 
                    t_face = target_faces[i]
                    if compare_faces(face , t_face , 0.7 ) : 
                        # print("Swapping faces ")
                        temp_frame = swap_face(
                            face_swapper,
                            face ,
                            reference_faces[i],
                            temp_frame
                        )
                        # return 
                        break 
        result = temp_frame   
        if restore : 
            result = bg_sampler(result )            
    else:
        result = frame 
        
    result =  cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # print("Restoring face ....  .. .. ")
    
    
    return result 

def parse_args():
    parser = argparse.ArgumentParser(description="Face swap.")
    parser.add_argument("--source_img", type=str, required=True, help="The path of source image, it can be multiple images, dir;dir2;dir3.")
    parser.add_argument("--target_video", type=str, required=True, help="The path of target image.")
    parser.add_argument("--output_video", type=str, required=False, default="result.png", help="The path and filename of output image.")
    parser.add_argument("--reference_img", type=str, required=True, help="The path of refernce image, it can be multiple images, dir;dir2;dir3.")
    parser.add_argument("--face_restore", action="store_true", help="The flag for face restoration.")
    parser.add_argument("--background_enhance", action="store_true", help="The flag for background enhancement.")
    parser.add_argument("--face_upsample", action="store_true", help="The flag for face upsample.")
    parser.add_argument("--upscale", type=int, default=1, help="The upscale value, up to 4.")
    parser.add_argument("--codeformer_fidelity", type=float, default=0.5, help="The codeformer fidelity.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    
    target_video_path = args.target_video
    
    target_img_paths = args.source_img.split(';')
    # print("Source image paths:", target_img_paths)
    
    reference_img_paths = args.reference_img.split(';') 
    # print("Reference image paths" , reference_img_paths)
    
   
    target_imgs = [Image.open(img_path) for img_path in target_img_paths]
    reference_imgs = [Image.open(img_path) for img_path in reference_img_paths]
  
    assert len(target_imgs ) == len(reference_imgs)  , print('number of target images should  be equal to the number of reference images ')

    # download from https://huggingface.co/deepinsight/inswapper/tree/main
    model = "./checkpoints/inswapper_128.onnx"
    result_image = process_video(target_video_path , target_img_paths, reference_img_paths, model)
        
