import matplotlib.pyplot as plt
import cv2
import torch
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
import numpy as np
import open3d as o3d


# Visualize the inlier matches with match count
def plot_matches(img1, img2, points1, points2, num_matches):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Create a canvas to draw both images side by side
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1, :] = (img1 * 255).astype(np.uint8)
    canvas[:h2, w1:, :] = (img2 * 255).astype(np.uint8)

    # Offset points for the second image
    points2_shifted = points2.copy()
    points2_shifted[:, 0] += w1

    # Plot matches
    plt.figure(figsize=(10, 5))
    plt.imshow(canvas)
    for (p1, p2) in zip(points1, points2_shifted):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='cyan', linewidth=0.5)
        plt.scatter(*p1, color='red', s=5)
        plt.scatter(*p2, color='blue', s=5)

    # Display number of matched points
    plt.text(10, 30, f'#Matches: {num_matches}', color='white', fontsize=12, 
             bbox=dict(facecolor='black', alpha=0.7))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Load images as torch.Tensors on GPU
    image0 = load_image('data/003.jpg').cuda()
    image1 = load_image('data/005.jpg').cuda()

    # Initialize the feature extractor and matcher
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # use SuperPoint+LightGlue
    matcher = LightGlue(features='superpoint').eval().cuda()

    # Extract features
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    # Match features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0
    points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1

    # Use RANSAC to filter out outliers
    H, mask = cv2.findHomography(points0, points1, cv2.RANSAC, 1.5)
    inliers0 = points0[mask.ravel() == 1]
    inliers1 = points1[mask.ravel() == 1]
    num_inliers = inliers0.shape[0]  # Number of inliers after RANSAC

    # Convert images back to numpy arrays for visualization
    image0_np = image0.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    image1_np = image1.cpu().numpy().transpose(1, 2, 0)

    # Call the plotting function with inliers only
    plot_matches(image0_np, image1_np, inliers0, inliers1, num_inliers)

    # 3D construction
    # Load camera parameters
    DISTORTION_COEFFICIENT = np.array([-0.0568965, 0, 0, 0, 0])
    MATCHING_RESULT = True
    ESSENTIAL_MATRIX_RESULT = True
    ROTATION_TRANSLATION_RESULT = True
    RECONSTRUCTION_TO_O3D = True
    fx, fy = 1086, 1086
    cx, cy = 512, 384
    k1 = -0.056896
    # Camera matrix K
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]])
    E, _ = cv2.findEssentialMat(points0, points1, K)
    print("Essential Matrix:\n", E)

    # c. The rotation and translation, R and t
    _, R, t, _ = cv2.recoverPose(E, points0, points1, K)
    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (t):\n", t)
    
    # normalized_keypoints1 = cv2.undistortPoints(points0.astype(np.float32), K, DISTORTION_COEFFICIENT).reshape(-1, 2)

    # normalized_keypoints2 = cv2.undistortPoints(points1.astype(np.float32), K, DISTORTION_COEFFICIENT).reshape(-1, 2)

    # P1 = np.eye(3, 4, dtype=np.float32)
    # P2 = np.hstack((R, t))
    # #! cv::triangulatePoints(): 모든 arguments는 float type으로 넣어주어야 한다.
    # X = cv2.triangulatePoints(P1, P2, normalized_keypoints1.T, normalized_keypoints2.T)
    # X /= X[3]
    # X = X[:3].T

    # print("* 3D points:", X, sep="\n", end="\n\n")

    # if RECONSTRUCTION_TO_O3D:
    #     # 3D reconstruction 결과 시각화 (Open3D)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(X)
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()
    #     vis.add_geometry(pcd)
    #     opt = vis.get_render_option()
    #     opt.point_size = 10.0  # Increase the point size
    #     vis.run()
    #     vis.destroy_window()
    # else:
        # import matplotlib.pyplot as plt

        # # 3D reconstruction 결과 시각화 (Matplotlib)            
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")

        # print(X)
        # ax.scatter(X[:, 0], X[:, 1], X[:, 2])
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # plt.show()