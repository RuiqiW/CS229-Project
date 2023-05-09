import open3d as o3d
import numpy as np
import os


VOXEL_DIMS = [36, 36, 8]

mesh_dir = '../data/12_meshMNIST/'
output_dir = '../data/12_voxel/'

for filename in os.listdir(mesh_dir):
    name, suffix = filename.split('.')
    print(name)
    if suffix != 'obj':
        continue  


    f = os.path.join(mesh_dir, filename)
    mesh = o3d.io.read_triangle_mesh(f)
    R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.02)

    # non-zero indices
    indices = [vox.grid_index-1 for vox in voxel_grid.get_voxels()]

    # voxel grid size
    size = np.max(np.vstack(indices), axis=0) + 1
    print(size)
    arr = np.zeros(size).astype(np.int8)
    for idx in indices:
        arr[tuple(idx)] = 1

    # pad voxel grid
    pad_width = [((VOXEL_DIMS[i] - size[i]) // 2, (VOXEL_DIMS[i] - size[i] + 1) // 2) for i in range(3)]
    voxels = np.pad(arr, pad_width)

    with open(os.path.join(output_dir, name + '.npy'), 'wb') as f:
        np.save(f, voxels)

