from math import cos, sin

import torch
from torch.hub import load_state_dict_from_url
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

from sixdrepnet.model import SixDRepNet
import sixdrepnet.utils as utils


class SixDRepNet_Detector():

    def __init__(self, gpu_id : int=0, dict_path: str=''):
        """
        Constructs the SixDRepNet instance with all necessary attributes.

        Parameters
        ----------
            gpu:id : int
                gpu identifier, for selecting cpu set -1
            dict_path : str
                Path for local weight file. Leaving it empty will automatically download a finetuned weight file.
        """

        self.gpu = gpu_id
        self.model = SixDRepNet(backbone_name='RepVGG-B1g2',
                                backbone_file='',
                                deploy=True,
                                pretrained=False,
                                gpu_id=self.gpu)
        # Load snapshot
        if dict_path=='':
            saved_state_dict = load_state_dict_from_url("https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth")    
        else:
            saved_state_dict = torch.load(dict_path)

        self.model.eval()
        self.model.load_state_dict(saved_state_dict)
        
        if self.gpu != -1:
            self.model.cuda(self.gpu)

        self.transformations = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def predict(self, img):
        """
        Predicts the persons head pose and returning it in euler angles.

        Parameters
        ----------
        img : array 
            Face crop to be predicted

        Returns
        -------
        pitch, yaw, roll
        """

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = self.transformations(img)

        img = torch.Tensor(img[None, :])

        if self.gpu != -1:
            img = img.cuda(self.gpu)
     
        pred = self.model(img)
                
        if self.gpu != -1:
            euler = utils.compute_euler_angles_from_rotation_matrices(pred)*180/np.pi
        else:
            euler = utils.compute_euler_angles_from_rotation_matrices(pred, False)*180/np.pi
        p = euler[:, 0].cpu().detach().numpy()
        y = euler[:, 1].cpu().detach().numpy()
        r = euler[:, 2].cpu().detach().numpy()

        return p,y,r


    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        """
        Prints the person's name and age.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        img : array
            Target image to be drawn on
        yaw : int
            yaw rotation
        pitch: int
            pitch rotation
        roll: int
            roll rotation
        tdx : int , optional
            shift on x axis
        tdy : int , optional
            shift on y axis
            
        Returns
        -------
        img : array
        """

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

        return img


