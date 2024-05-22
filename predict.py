# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from swapper import process_video  , create_video 
import matplotlib.pyplot as plt 
import insightface
import onnxruntime
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from realesrgan import RealESRGANer
import cv2 
from gfp.utils import GFPGANer 
import os 
import glob
import re
# improt 

class Predictor(BasePredictor):

    def getFaceSwapModel(self , model_path: str):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = insightface.model_zoo.get_model(model_path)
        return model


    def getFaceAnalyser(self):
        det_size=(320, 320)
        providers = onnxruntime.get_available_providers()
        # print("The provider is : " , providers)
        face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
        face_analyser.prepare(ctx_id=0, det_size=det_size)
        return face_analyser 
    
    def getFaceEnhancer(self) : 
        # bg_tile = 400 
        # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        # GFPGan version default = 1.3 
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        # url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    
        model_path = os.path.join('./gfp/weights/' +  model_name + '.pth')
        restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier )

        return restorer 
    
    def create_video(self , h , w  ,files  ) : 
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output1.mp4", fourcc,25, (w, h))
        # print("Wahts jmwifndfvnio")
        # Write each image to the video
        for image in files:
                image = cv2.imread("images/"+image)
                out.write(image)
        out.release()

    def setup(self) -> None:
        self.model = "./checkpoints/inswapper_128_fp16.onnx"
        self.analyser = self.getFaceAnalyser()
        self.swapper = self.getFaceSwapModel(self.model) 
        self.enhancer = self.getFaceEnhancer()
        

    def predict(
        self,
        target_image: Path = Input(description="Target input image"),
        refer_image : Path = Input(description = "Reference input image"), 
        video : Path = Input(description = "Video input ")
        ) -> Path:
        result_image = process_video(video, target_image , refer_image , self.model  , restore = True , bg_upsampler = False  , face_analyser=self.analyser , face_swapper=self.swapper , restorer=self.enhancer )
    
        files = glob.glob('images/*.png')
        files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))
        files = [os.path.basename(p) for p in files]

        h ,w , _ = cv2.imread("./images/0.png").shape 
    
        create_video(h, w , files )

        return './ouptut.mp4'
    
    
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
