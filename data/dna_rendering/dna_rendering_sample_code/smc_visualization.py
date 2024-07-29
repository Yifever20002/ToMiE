from SMCReader import SMCReader
import numpy as np
import os
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.transform.camera.distortion import undistort_images
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.core.visualization.visualize_keypoints3d import visualize_project_keypoints3d
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_calibration

import argparse

def options():
    parser = argparse.ArgumentParser()
    ############ necessary data paths
    parser.add_argument("--smc_main_file", help="path to smc file which contains RGB images", type=str, required=True)
    parser.add_argument("--smc_annots_file", help="path to smc file which contains SMPLx parameters", type=str, required=True)
    parser.add_argument("--body_models", help="path to smplx base model", type=str, default="./data/body_models")
    parser.add_argument("--out_dir", help="output folder", type=str, default="./vis_smc")
    args = parser.parse_args()
    return args  

class SMCVisualizer:

    def __init__(self, file_path=None, annots_file_path=None, smc_reader=None, smc_annots_reader=None):
        """Visualize SMPL and Keypoints from SMC file or Initialized SMCReader instance.
        Args:
            file_path (str):
                Path to an SMC file.
            smc_reader (SMCReader):
                Initialized SMCReader instance.
        """
        assert (file_path is not None or smc_reader is not None)
        assert (annots_file_path is not None or smc_annots_reader is not None)
        if smc_reader is not None:
            self.smc_reader = smc_reader
        elif file_path is not None:
            self.smc_reader = SMCReader(file_path)
        else:
            raise ValueError('file_path and smc_reader cannot be None at the same time.')
        
        if smc_annots_reader is not None:
            self.smc_annots_reader = smc_annots_reader
        elif file_path is not None:
            self.smc_annots_reader = SMCReader(annots_file_path)
        else:
            raise ValueError('file_path and smc_reader cannot be None at the same time.')

    def visualize_smpl(self, Camera_id, output_dir, Frame_id=None, visualize_kps=True, smpl_model_path='mmhuman3d/data/body_models', disable_tqdm=True):
        '''
            Camera_id: int
            output_dir: str
            Frame_id: List[int]
            visualize_kps: bool, default=True
            smpl_model_path: str, default='mmhuman3d/data/body_models'
            disable_tqdm: bool, default=True
        '''
        image_array = self.smc_reader.get_img('Camera_5mp', Camera_id, Image_type='color', Frame_id=Frame_id, disable_tqdm=disable_tqdm)
        if len(image_array.shape) == 3:
            image_array = image_array[None]
        cam_params = self.smc_annots_reader.get_Calibration(Camera_id)
        Camera_id = str(Camera_id)
        camera_parameter = FisheyeCameraParameter(name=Camera_id)
        K = cam_params['K']
        D = cam_params['D'] # k1, k2, p1, p2, k3
        RT = cam_params['RT']
        R = RT[:3, :3]
        T = RT[:3, 3]
        corrected_img = image_array
        # extrinsic = cam_params['RT']
        # r_mat_inv = extrinsic[:3, :3]
        # r_mat = np.linalg.inv(r_mat_inv)
        # t_vec = extrinsic[:3, 3:]
        # t_vec = -np.dot(r_mat, t_vec).reshape((3))
        # R = r_mat
        # T = t_vec

        # dist_coeff_k = [D[0],D[1],D[4]]
        # dist_coeff_p = D[2:4]
        # camera_parameter.set_KRT(K, R, T)
        # camera_parameter.set_dist_coeff(dist_coeff_k, dist_coeff_p)
        # camera_parameter.inverse_extrinsic()
        # camera_parameter.set_resolution(image_array.shape[1], image_array.shape[2])

        # corrected_cam, corrected_img = undistort_images(camera_parameter, image_array)
        # K = np.asarray(corrected_cam.get_intrinsic())
        # R = np.asarray(corrected_cam.get_extrinsic_r())
        # T = np.asarray(corrected_cam.get_extrinsic_t())

        smpl_dict = self.smc_annots_reader.get_SMPLx(Frame_id=Frame_id)

        fullpose = smpl_dict['fullpose']
        betas = smpl_dict['betas']
        transl = smpl_dict['transl']

        if len(fullpose.shape) == 2:
            fullpose = fullpose[None]
        if len(betas.shape) == 1:
            betas = betas[None]
        if len(transl.shape) == 1:
            transl = transl[None]

        fullpose = fullpose.reshape(fullpose.shape[0], -1)

        gender = self.smc_reader.actor_info['gender']
        body_model=dict(
            type='SMPLX',
            gender=gender,
            num_betas=10,
            keypoint_convention='smplx', 
            model_path=smpl_model_path,
            batch_size=1,
            use_face_contour=True,
            use_pca=False,
            num_pca_comps=24,
            flat_hand_mean=False)

        results = visualize_smpl_calibration(
                poses=fullpose,
                betas=betas,
                transl=transl,
                K=K,
                R=R,
                T=T,
                overwrite=True,
                body_model_config=body_model,
                output_path=output_dir, 
                image_array=corrected_img,
                resolution=(corrected_img.shape[1], corrected_img.shape[2]),
                return_tensor=True,
                alpha=0.8,
                batch_size=1,
                plot_kps=True,
                vis_kp_index=False)
        
if __name__ == '__main__':
    args = options()
    os.makedirs(args.out_dir, exist_ok=True)

    visualizer = SMCVisualizer(file_path=args.smc_main_file, annots_file_path=args.smc_annots_file)
    visualizer.visualize_smpl(Camera_id=25, output_dir=args.out_dir, Frame_id=list(range(0, 40, 4)), smpl_model_path=args.body_models)
    print("=== done", flush=True)
