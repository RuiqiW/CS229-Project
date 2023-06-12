import open3d as o3d
import os
import numpy as np
import trimesh




# RAND = True
PREFIX = 'train'
mesh_dir = '../{}_meshMNIST/train/'.format(PREFIX)
output_dir = '../{}_mesh/train/'.format(PREFIX)
# rot_mat_dir = '../data/{}_rotations'.format(PREFIX)


idx = 0

label_file = os.path.join('../train_meshMNIST', 'labels.txt')
f = open(label_file, 'r')
lines = f.readlines()

count = np.zeros(10)

num_faces = 0
for filename in os.listdir(mesh_dir):
    name, suffix = filename.split('.')
    
    if idx % 1000 == 0:
        print(idx)
        print(count)

    if suffix != 'obj':
        continue  

    f = os.path.join(mesh_dir, filename)

    # mesh = trimesh.load(f)
    # F = np.array(mesh.faces).shape[0]
    # print(F)
    # faces = trimesh.repair.broken_faces(mesh)
    # print(len(faces))
    # face_mask = np.ones(F, dtype=bool)
    # for i in faces:
    #     face_mask[i] = False
    # mesh.update_faces(face_mask)

    # print(np.array(mesh.faces).shape[0])
    # trimesh.repair.fill_holes(mesh)
    # # print(mesh.is_watertight)


    mesh = o3d.io.read_triangle_mesh(f)
    mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=480)
    if not mesh_smp.is_watertight():
        id = int(name)
        label = int(lines[id-1].split(',')[1])
        count[label] += 1
        continue

    # # print(
    # #     f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
    # # )
    # # o3d.visualization.draw_geometries([mesh_smp])
    o3d.io.write_triangle_mesh(os.path.join(output_dir, name + '.obj'), mesh_smp)

    idx += 1

print(count)