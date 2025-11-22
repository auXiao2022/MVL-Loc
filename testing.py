import numpy as np
import quaternion  # 需要安装 numpy-quaternion 库
import scipy.spatial.transform as tf
import numpy as np
import os
import torch
from torch import nn
import scipy.linalg as slin
import math
import transforms3d.quaternions as txq
import transforms3d.euler as txe
# def quaternion_to_log(quat):
#     # 将四元数转换为旋转矩阵
#     rotation = tf.Rotation.from_quat(quat)
#     # 提取旋转矢量（对数四元数）
#     log_quat = rotation.as_rotvec()
#     return log_quat


# q_4 = [-0.0964216,0.016434999999999998,0.006233600000000001,0.9951853999999999]
# q_4 = np.array(q_4)
# q_3 = quaternion_to_log(q_4)
# print(q_3)

# R = np.array([[0.99935108, -0.01557608,  0.03150894],
#        [ 0.00923751,  0.98130137,  0.19211653],
#        [-0.03391284, -0.19170459,  0.98083067]])


# # 将旋转矩阵转换为 Rotation 对象
# rotation = tf.Rotation.from_matrix(R)

# # 将 Rotation 对象转换为四元数
# quat = rotation.as_quat()
# # print(quat)

# array([-0.09642176,  0.01643496,  0.00623355,  0.99518535])

import numpy as np
import scipy.spatial.transform as tf
import transforms3d.quaternions as txq
# 给定的四元数
# quat = np.array([-0.09642176, 0.01643496, 0.00623355, 0.99518535])
quat = np.array([0.9951853999999999, -0.0964216, 0.016434999999999998, 0.006233600000000001],dtype=np.float64)
R = np.array([[0.99935108, -0.01557608,  0.03150894],
       [ 0.00923751,  0.98130137,  0.19211653],
       [-0.03391284, -0.19170459,  0.98083067]],dtype=np.float64)
# np.dot(align_R,

q = np.array([-0.096577549567277057,0.016461546921178523,0.006243617133769956],dtype=np.float64)
q_test = txq.mat2quat(R)
print("q_test",q_test)
def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q

q_test1 = qlog(quat)
print("q_test1",q_test1)
# mat_test = txq.quat2mat(q)
# print("mat",mat_test)
# 将旋转矩阵转换为 Rotation 对象
# rotation = tf.Rotation.from_matrix(R)

# # 将 Rotation 对象转换为四元数
# quat = rotation.as_quat()
# # 将四元数转换为 Rotation 对象
# rotation1 = tf.Rotation.from_quat(quat)


# 将 Rotation 对象转换为旋转矩阵
# R1 = rotation1.as_matrix()

# print(R1)

# def quaternion_to_log(quat):
#     # 将四元数转换为 Rotation 对象
#     rotation = tf.Rotation.from_quat(quat)
    
#     # 确保四元数位于同一半球
#     if quat[0] < 0:
#         quat = -quat
    
#     # 将 Rotation 对象转换为旋转矢量（对数四元数）
#     log_quat = rotation.as_rotvec()
    
#     return log_quat

# # 示例四元数
# quat = np.array([-0.09642176, 0.01643496, 0.00623355, 0.99518535])

# # 将四元数转换为对数四元数并限制在同一半球
# log_quat = quaternion_to_log(quat)

# print("Logarithm of Quaternion:", log_quat)

# def qlog(q):
#     if all(q[1:] == 0):
#         q = np.zeros(3)
#     else:
#         q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
#     return q
# def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
#     poses_out = np.zeros((len(poses_in), 6))
#     poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]
    
#   # align
#     for i in range(len(poses_out)):
#         R = poses_in[i].reshape((3, 4))[:3, :3]
#         q = txq.mat2quat(np.dot(align_R, R))
#         q *= np.sign(q[0])  # constrain to hemisphere
#         q = qlog(q)
#         poses_out[i, 3:] = q
#         t = poses_out[i, :3] - align_t
#         poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()
        
# poses_out = np.zeros(-1, 6) 
# poses_out[:, 0:3] = R[:, [3, 7, 11]]