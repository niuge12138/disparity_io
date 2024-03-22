import numpy as np
import cv2
import open3d as o3d


def disp2grey_16bit(disp, filepath):
    """
    convert disparity to grey map
    save to 16bit grey map
    disp: numpy array, shape [1, H, W] or [H, W]
    filepath: .png save path
    """
    max_disp, min_disp = np.max(disp), np.min(disp)
    scale_disp = ((disp - min_disp) / (max_disp - np.min(disp)) * 65525).astype(np.uint16)
    cv2.imwrite(filepath, scale_disp, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def disp2grey_8bit(disp, filepath):
    """
    convert disparity to grey map
    save to 8bit grey map
    disp: numpy array, shape [1, H, W] or [H, W]
    filepath: .png save path
    """
    max_disp, min_disp = np.max(disp), np.min(disp)
    scale_disp = ((disp - min_disp) / (max_disp - np.min(disp)) * 255).astype(np.uint8)
    cv2.imwrite(filepath, scale_disp)


def disp2color(disp, filepath):
    """
    convert disparity to pseudo color map
    disp: numpy array, shape [1, H, W] or [H, W]
    filepath: .png save path
    """
    disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=255/disp.max(), beta=0), cv2.COLORMAP_JET)
    cv2.imwrite(filepath, disp_color)


def disp2pc(disp, image_rgb, b, f, cx, cy, plysavepth, maxdepth=30):
    """
    convert disparity to point cloud
    disp: disparity
    image_rgb: rgb image (H, W, 3) with channel order R-G-B
    b: binocular camara baseline length, m
    f: focal length, pixel
    cx cy: optical center coordinate, pixel
    plysavepth: .ply file save path
    maxdepth: maximum depth value
    """
    Z = b * f / disp
    H, W = Z.shape

    # step1: build camera intrinsic matrix
    K = np.array([[f, 0, cx,], [0, f, cy], [0, 0, 1]])

    # step2
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv_points = np.vstack((u.flatten(), v.flatten(), np.ones_like(u.flatten())))
    normalized_plane_coordinates = np.linalg.inv(K) @ uv_points
    
    # step3: calculate 3D coordinate
    X_norm = normalized_plane_coordinates[0, :]
    Y_norm = normalized_plane_coordinates[1, :]
    Z_norm = Z.flatten()
    
    X = Z_norm * X_norm
    Y = Z_norm * Y_norm
    Z = Z.flatten()
    
    # reshape to (H, W, 3)
    depth_map = np.stack((X, Y, Z), axis=-1)
    mask = (depth_map[:, 2] <= maxdepth) & (depth_map[:, 2] > 0)
    depth_map = depth_map[mask]
    
    color = image_rgb.reshape(-1, 3)
    color = color[mask]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(depth_map)
    point_cloud.colors = o3d.utility.Vector3dVector(color / 255.)
    
    o3d.io.write_point_cloud(plysavepth, point_cloud)
    print(".pfm is transformed to .ply successfully !")


def disp2ply_Q(disp, image_rgb, Q, plysavepth, maxdepth=30):
    """
    disp: disparity
    image_rgb: rgb image (H, W, 3) with channel order R-G-B
    Q: Q matrix
    """
    depth_map = cv2.reprojectImageTo3D(disp.astype(np.float32), Q)
    point_cloud = o3d.geometry.PointCloud()
    depth_map = depth_map.reshape(-1, 3)
    mask = (depth_map[:, 2] <= maxdepth) & (depth_map[:, 2] > 0)
    depth_map = depth_map[mask]
    color = image_rgb.reshape(-1, 3)
    color = color[mask]
    point_cloud.points = o3d.utility.Vector3dVector(depth_map)
    point_cloud.colors = o3d.utility.Vector3dVector(color / 255.)
    o3d.io.write_point_cloud(plysavepth, point_cloud)
    print(".pfm is transformed to .ply successfully !")


# if __name__ == "__main__":
