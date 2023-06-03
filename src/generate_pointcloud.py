import open3d as o3d
import os
import numpy as np



RAND = True
PREFIX = 'train'
mesh_dir = '../data/{}_meshMNIST/'.format(PREFIX)
output_dir = '../data/{}_pcd'.format(PREFIX)
rot_mat_dir = '../data/{}_rotations'.format(PREFIX)


idx = 0
# for filename in ['00001.obj']:
for filename in os.listdir(mesh_dir):
    name, suffix = filename.split('.')
    
    if idx % 1000 == 0:
        print(idx)

    if suffix != 'obj':
        continue  

    f = os.path.join(mesh_dir, filename)
    mesh = o3d.io.read_triangle_mesh(f)

    if RAND:
        # axis = (0, np.random.rand() * 360, 0)
        # R = mesh.get_rotation_matrix_from_xyz(axis)
        R = np.load(os.path.join(rot_mat_dir, name + ".npy"))
        mesh.rotate(R, center=mesh.get_center())

    # o3d.visualization.draw_geometries([mesh])

    pcd = mesh.sample_points_uniformly(number_of_points=2500)
    pcd = mesh.sample_points_poisson_disk(number_of_points=500, pcl=pcd)

    # o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud(os.path.join(output_dir, name + '.ply'), pcd, write_ascii=True)

    idx += 1