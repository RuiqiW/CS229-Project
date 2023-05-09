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



mesh_dir = '../data/12_meshMNIST/'
output_dir = '../data/12_single_view' if RAND else '../data/12_proj/'

renderer = pyrender.OffscreenRenderer(W, H)
pyrender_camera = pyrender.camera.OrthographicCamera(1.0, 1.0)

T = np.identity(4)
T[0:3, 0:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
T[0:3, 3] = np.array([0.5, 0.5, -1])


for filename in os.listdir(mesh_dir):
    name, suffix = filename.split('.')
    print(name)
    if suffix != 'obj':
        continue  
    
    f = os.path.join(mesh_dir, filename)

    tm = trimesh.load(f)
    mesh = pyrender.Mesh.from_trimesh(tm)

    # Creates the scene and adds the mesh.
    scene = pyrender.Scene()
    node = scene.add(mesh)

    if RAND:
        # Generate random pose
        q = np.random.randn(4)
        q /= np.linalg.norm(q)
        R = qvec2rotmat(q)
        rot = np.identity(4)
        rot[0:3, 0:3] = R

        t1 = np.identity(4)
        t1[0:3, 3] = -mesh.centroid
        t2 = np.identity(4)
        t2[0:3, 3] = mesh.centroid

        pose = t2 @ rot @ t1
        scene.set_pose(node, pose)

    
    cam_node = scene.add(pyrender_camera, pose=T)

    # pyrender.Viewer(scene, use_raymond_lighting=True)

    color, depth = renderer.render(scene)
    # depth = np.asarray(depth)
    # plt.imshow(color[30:86, 30:86, :])
    # plt.show()
    plt.imsave(os.path.join(output_dir, name + ".png"), color[30:86, 30:86, :], cmap="gray")
    plt.close()

    del scene




