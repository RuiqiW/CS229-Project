import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import pandas as pd
import argparse

import cv2
import os


W, H = 112, 112
RAND = True

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def axis_angle_to_quaternion(axis, angle):
    axis = np.array(axis)
    # axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(angle)
    
    qw = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    quaternion = np.array([qw, xyz[0], xyz[1], xyz[2]])
    
    return quaternion


PREFIX = 'train'
mesh_dir = '../data/{}_meshMNIST/'.format(PREFIX)
output_dir = '../data/{}_multi_view'.format(PREFIX)
rot_mat_dir = '../data/{}_rotations'.format(PREFIX)

renderer = pyrender.OffscreenRenderer(W, H)
pyrender_camera = pyrender.camera.OrthographicCamera(1.0, 1.0)

T = np.identity(4)
T[0:3, 0:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
T[0:3, 3] = np.array([0.5, 0.5, -1])


rot_matrices = []

for axis in [[1, 0, 0], [0, 1, 0]]:
    for angle in [0, 90, 180, 270]:
        qvec = axis_angle_to_quaternion(axis, angle)
        R = qvec2rotmat(qvec)
        rot = np.identity(4)
        rot[0:3, 0:3] = R
        rot_matrices.append(rot)


idx = 0
# for filename in ['00001.obj']:
for filename in os.listdir(mesh_dir):
    name, suffix = filename.split('.')
    # print(name)
    if idx % 1000 == 0:
        print(idx)
        
    if suffix != 'obj':
        continue  
    
    f = os.path.join(mesh_dir, filename)

    tm = trimesh.load(f)
    mesh = pyrender.Mesh.from_trimesh(tm)

    # Creates the scene and adds the mesh.
    scene = pyrender.Scene()
    node = scene.add(mesh)


    t1 = np.identity(4)
    t1[0:3, 3] = -mesh.centroid
    t2 = np.identity(4)
    t2[0:3, 3] = mesh.centroid

    if RAND:
        # Generate random pose
        q = np.random.randn(4)
        q /= np.linalg.norm(q)
        R = qvec2rotmat(q)
        rot = np.identity(4)
        rot[0:3, 0:3] = R

        np.save(os.path.join(rot_mat_dir, name + ".npy"), R)

    else:
        angle = np.random.rand() * 360

        qvec = axis_angle_to_quaternion([0, 1, 0], angle)
        R = qvec2rotmat(qvec)
        rot = np.identity(4)
        rot[0:3, 0:3] = R
        
    
    cam_node = scene.add(pyrender_camera, pose=T)

    images = []
    for i in range(len(rot_matrices)):
        R = rot_matrices[i]
        if RAND:
            pose = t2 @ R @ rot @ t1
        else:
            pose = t2 @ R @ rot @ t1
        scene.set_pose(node, pose)

        # pyrender.Viewer(scene, use_raymond_lighting=True)

        color, depth = renderer.render(scene)

        images.append(color[30:86, 30:86, 0])
        # depth = np.asarray(depth)
        # plt.imshow(color[30:86, 30:86, 0])
        # plt.show()
        # plt.imsave(os.path.join(output_dir, name + "_{:02d}".format(i) + ".png"), color[30:86, 30:86, :], cmap="gray")
        # plt.close()

    res = np.stack(images, axis=0)
    # print(res.shape)
    np.save(os.path.join(output_dir, name + ".npy"), res)

    del scene

    idx += 1
