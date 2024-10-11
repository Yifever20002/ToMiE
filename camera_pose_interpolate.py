import numpy as np
import glob
import os 
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d


# 固定点求解函数
def find_fixed_point(camera_matrices):
  # 保存相机位置和平移向量
  origins = camera_matrices[:, :3, 3]  # 提取平移向量（相机位置）
  
  # 提取旋转矩阵中的z轴方向（相机的视线方向）
  directions = -camera_matrices[:, :3, 2]  # 取z轴作为相机视线方向
  
  # 使用最小二乘法求解视线交点
  A = np.zeros((len(directions), 3))
  b = np.zeros(len(directions))
  
  for i, (origin, direction) in enumerate(zip(origins, directions)):
    direction = direction / np.linalg.norm(direction)  # 归一化视线方向
    A[i] = direction
    b[i] = np.dot(origin, direction)
  
  # 求解最小二乘法，计算交点
  fixed_point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
  
  return fixed_point


# 读取48.txt到59.txt的相机位姿
files = glob.glob('render360/*.txt')
files = [f for f in files if 48 <= int(os.path.basename(f).split('.')[0]) <= 59]
files = sorted(files)

camera_matrices = []
for file in files:
  matrix = np.loadtxt(file)
  camera_matrices.append(matrix)

camera_matrices = np.array(camera_matrices)

# 计算相机共同的固定点
fixed_point = find_fixed_point(camera_matrices)
print("固定点的坐标:", fixed_point)

# 提取旋转部分（3x3矩阵）和平移部分（3x1向量）
rotations = camera_matrices[:, :3, :3]
translations = camera_matrices[:, :3, 3]

# 计算相机的视角方向，方向为固定点 - 相机的平移位置
view_directions = fixed_point - translations

# 将旋转矩阵转换为四元数
rot_obj = R.from_matrix(rotations)

# 在球面上进行旋转插值（Slerp）
key_times = np.arange(len(rot_obj))  # 原有的相机位姿对应的时间点
interp_times = np.linspace(0, len(rot_obj) - 1, 100)  # 要插值生成的100个新时间点

slerp = Slerp(key_times, rot_obj)
interp_rotations = slerp(interp_times)

# 对平移部分进行线性插值
interp_translations = interp1d(key_times, translations, axis=0)(interp_times)

# 确保相机位置绕着固定点
for i, (rotation, translation) in enumerate(zip(interp_rotations, interp_translations)):
  new_camera_matrix = np.eye(4)
  new_camera_matrix[:3, :3] = rotation.as_matrix()  # 使用插值后的旋转矩阵
  
  # 重新计算平移部分，确保相机始终看向固定点
  direction = fixed_point - translation  # 相机指向固定点的方向
  direction = direction / np.linalg.norm(direction)  # 归一化方向
  
  # 设置新的平移部分为原来的相机位置，但保证方向始终看向固定点
  new_camera_matrix[:3, 3] = translation  # 使用插值后的相机位置
  
  # 保存插值后的相机矩阵
  np.savetxt(f'render360_dense/{i:03d}.txt', new_camera_matrix, fmt='%.6f')
