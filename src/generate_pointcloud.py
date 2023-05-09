import open3d as o3d
import os



mesh_dir = 'data/12_meshMNIST/'
output_dir = 'data/12_pcd/'


for filename in os.listdir(mesh_dir):
    name, suffix = filename.split('.')
    print(name)
    if suffix != 'obj':
        continue  

    f = os.path.join(mesh_dir, filename)
    mesh = o3d.io.read_triangle_mesh(f)

    pcd = mesh.sample_points_uniformly(number_of_points=2500)
    pcd = mesh.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
    # o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud(os.path.join(output_dir, name + '.ply'), pcd, write_ascii=True)