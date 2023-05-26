import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size, radius_normal_factor, radius_feature_factor):

    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * radius_normal_factor

    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * radius_feature_factor
    
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


if __name__ == '__main__':
    # voxel_size: 0.02
    # radius_normal_factor
    # radius_feature_factor

    pcd = o3d.io.read_point_cloud('../data/train_pcd/00003.ply')
    pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size=0.02, radius_normal_factor=2, radius_feature_factor=2)

    # print(np.asarray(pcd_down.points).shape)
    print(pcd_fpfh.data.shape)
    # o3d.visualization.draw_geometries([pcd_down])
