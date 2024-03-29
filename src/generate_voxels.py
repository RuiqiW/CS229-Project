import open3d as o3d
import numpy as np
import os


RAND = True
VOXEL_DIMS = [44, 44, 44] if RAND else [36, 36, 36]

PREFIX = 'train'
mesh_dir = '../data/{}_meshMNIST/'.format(PREFIX)
output_dir = '../data/{}_voxel'.format(PREFIX)
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
    R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))

    if RAND:
        R_rand = np.load(os.path.join(rot_mat_dir, name + ".npy"))
        mesh.rotate(R_rand @ R, center=mesh.get_center())
    else:
        mesh.rotate(R, center=mesh.get_center())

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.02)
    
    # o3d.visualization.draw_geometries([voxel_grid])

    # non-zero indices
    indices = [vox.grid_index-1 for vox in voxel_grid.get_voxels()]

    # voxel grid size
    size = np.max(np.vstack(indices), axis=0) + 1
    arr = np.zeros(size).astype(np.int8)
    for grid_idx in indices:
        arr[tuple(grid_idx)] = 1

    for i in range(3):
        if size[i] > VOXEL_DIMS[i]:
            start = (size[i] - VOXEL_DIMS[i]) // 2
            end = -(size[i] - VOXEL_DIMS[i] - start)

            print(start, end)

            if i == 0:
                arr = arr[start:end, : , :]
            elif i == 1:
                arr = arr[:, start:end, :]
            elif i == 2:
                arr = arr[:, :, start:end]


    size = np.clip(size, 0, VOXEL_DIMS)

    # pad voxel grid
    pad_width = [((VOXEL_DIMS[i] - size[i]) // 2, (VOXEL_DIMS[i] - size[i] + 1) // 2) for i in range(3)]
    voxels = np.pad(arr, pad_width)

    with open(os.path.join(output_dir, name + '.npy'), 'wb') as f:
        np.save(f, voxels)

    idx += 1

