from calendar import c
from functools import partial
import json
from unittest.mock import NonCallableMagicMock

import cv2
import h5py
import numpy as np
import tqdm
import argparse
import os 


class SMCReader:

    def __init__(self, file_path):
        """Read SenseMocapFile endswith ".smc".

        Args:
            file_path (str):
                Path to an SMC file.
            body_model (nn.Module or dict):
                Only needed for SMPL transformation to device frame
                if nn.Module: a body_model instance
                if dict: a body_model config
        """
        self.smc = h5py.File(file_path, 'r')
        self.__calibration_dict__ = None

    ### Calibration
    def get_Calibration_all(self):
        """Get calibration matrix of all cameras and save it in self
        
        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict( 
                Camera_group: Camera_id : Matrix_type : value
              )
            Notice:
                Camera_group(str) in ['Camera_12mp', 'Camera_5mp','Kinect']
                Camera_id(str) in {'Camera_5mp': '0'~'47',  'Camera_12mp':'48'~'60',
                    'Kinect': '0'~'7'}
                Matrix_type in ['D', 'K', 'R', 'RT', 'T'] 
                ###TODO definition of diffent Matrix_type###
        """  
        if self.__calibration_dict__ is not None:
            return self.__calibration_dict__

        self.__calibration_dict__ = dict()
        for ci in self.smc['Camera_Parameter'].keys():
            self.__calibration_dict__.setdefault(ci,dict())
            for mt in ['D', 'K', 'RT', 'Color_Calibration'] :
                self.__calibration_dict__[ci][mt] = \
                    self.smc['Camera_Parameter'][ci][mt][()]
        return self.__calibration_dict__

    def get_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain camera by its type and id 

        Args:
            Camera_group (str):
                Camera_group in ['Camera_12mp', 'Camera_5mp','Kinect'].
            Camera_id (int/str of a number):
                CameraID(str) in {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60',
                    'Kinect': '0'~'7'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'R', 'RT', 'T'] 
        """
        rs = dict()
        for k in ['D', 'K', 'RT', 'Color_Calibration'] :
            rs[k] = self.smc['Camera_Parameter'][f'{int(Camera_id):02d}'][k][()]
        return rs

    ### RGB image
    def __read_color_from_bytes__(self, color_array):
        """Decode an RGB image from an encoded byte array."""
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    def get_mask(self, Camera_id, Frame_id=None,disable_tqdm=True):
        """Get image its Camera_group, Camera_id, Image_type and Frame_id

        Args:
            Camera_group (str):
                Camera_group in ['Camera_12mp', 'Camera_5mp','Kinect'].
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60',
                    'Kinect': '0'~'7'}
            Image_type(str) in 
                    {'Camera_5mp': ['color','mask'],  
                    'Camera_12mp': ['color','mask'],
                    'Kinect': ['depth', 'mask']}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame'('149') 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
              'color': HWC in bgr (uint8)
              'mask' : HW (uint8)
              'depth': HW (uint16)
        """ 
        Camera_id = str(Camera_id)

        assert(isinstance(Frame_id,(list,int, str, type(None))))
        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert(Frame_id in self.smc['Mask'][Camera_id]['mask'].keys())
            img_byte = self.smc['Mask'][Camera_id]['mask'][Frame_id][()]
            img_color = self.__read_color_from_bytes__(img_byte)
            img_color = np.max(img_color,2)
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list =sorted([int(l) for l in self.smc['Mask'][Camera_id]['mask'].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_mask(Camera_id,fi))
            return np.stack(rs,axis=0)
    
    ###Keypoints2d
    def get_Keypoints2d(self, Camera_id,Frame_id=None):
        """Get keypoint2D by its Camera_group, Camera_id and Frame_id

        Args:
            Camera_group (str):
                Camera_group in ['Camera_12mp', 'Camera_5mp'].
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60',}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'('149') 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
              'color': HWC in bgr (uint8)
              'mask' : HW (uint8)
              'depth': HW (uint16)
        """ 
        Camera_id = f'{int(Camera_id):02d}'
        assert(isinstance(Frame_id,(list,int, str, type(None))))
        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            return self.smc['Keypoints_2D'][Camera_id][()][Frame_id,:]
        else:
            if Frame_id is None:
                return self.smc['Keypoints_2D'][Camera_id][()]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list):
                rs.append(self.get_Keypoints2d(Camera_id,fi))
            return np.stack(rs,axis=0)

    ###Keypoints3d
    def get_Keypoints3d(self, Frame_id=None):
        """Get keypoint3D Frame_id, TODO coordinate

        Args:
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'('149') 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            Keypoints3d tensor: np.ndarray of shape ([N], ,3)
        """ 
        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            return self.smc['Keypoints_3D'][Frame_id,:]
        else:
            if Frame_id is None:
                return self.smc['Keypoints_3D']
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list):
                rs.append(self.get_Keypoints3d(fi))
            return np.stack(rs,axis=0)

    ###SMPLx
    def get_SMPLx(self, Frame_id=None):
        """Get SMPL (world coordinate) computed by mocap processing pipeline.

        Args:
            Frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            dict:
                'global_orient': np.ndarray of shape (N, 3)
                'body_pose': np.ndarray of shape (N, 21, 3)
                'transl': np.ndarray of shape (N, 3)
                'betas': np.ndarray of shape (1, 10)
        """
        t_frame = self.smc['SMPLx']['betas'][()].shape[0]
        if Frame_id is None:
            frame_list = range(t_frame)
        elif isinstance(Frame_id, list):
            frame_list = [int(fi) for fi in Frame_id]
        elif isinstance(Frame_id, (int,str)):
            Frame_id = int(Frame_id)
            assert Frame_id < t_frame,\
                f'Invalid frame_index {Frame_id}'
            frame_list = Frame_id
        else:
            raise TypeError('frame_id should be int, list or None.')

        smpl_dict = {}
        for key in ['betas', 'expression', 'fullpose', 'transl']:
            smpl_dict[key] = self.smc['SMPLx'][key][()][frame_list, ...]
        smpl_dict['scale'] = self.smc['SMPLx']['scale'][()]

        return smpl_dict

def options():
    parser = argparse.ArgumentParser()
    ############ necessary data paths
    parser.add_argument("--smc_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./dna_rendering_data")

    args = parser.parse_args()
    return args  

### test func
if __name__ == '__main__':
    args = options()
    smc_name = args.smc_file.split('/')[-1].split('.')[0]
    outdir = os.path.join(args.outdir, smc_name)
    os.makedirs(outdir, exist_ok=True)

    rd = SMCReader(args.smc_file)
    # print('mask:\n', rd.get_mask(6, 10))
    # print('camera parameters:\n', rd.get_Calibration(20))
    # print('kp2d:\n', rd.get_Keypoints2d(15))
    # print('kp3d:\n', rd.get_Keypoints3d()["keypoints3d"])
    # print('smplx:\n', rd.get_SMPLx())
    smplx = rd.get_SMPLx()
    for k, v in smplx.items():
        print(k, v.shape)

