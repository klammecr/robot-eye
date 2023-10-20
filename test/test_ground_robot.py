# Third Party
import pytest
from pyquaternion import Quaternion
import numpy as np

# In House
from roboteye.ground_robot import GroundRobot, Frames, COB
from roboteye.utils.test_utils import get_calib, create_bbox_pts

# Constants
IMG_SIZE = (256, 512)

# Default Calibration Constants
cam_dx = 0.5
cam_dy = 0.
cam_dz = 0.25

def rotationMatrixToQuaternion1(m):
    #q0 = qw
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if(t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2,1] - m[1,2]) * t
        q[1] = (m[0,2] - m[2,0]) * t
        q[2] = (m[1,0] - m[0,1]) * t

    else:
        i = 0
        if (m[1,1] > m[0,0]):
            i = 1
        if (m[2,2] > m[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3

        t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k,j] - m[j,k]) * t
        q[j] = (m[j,i] + m[i,j]) * t
        q[k] = (m[k,i] + m[i,k]) * t

    return q

class TestGroundRobot:
    def test_init(self):
        calib = get_calib(img_size=IMG_SIZE, cam_dx=cam_dx, cam_dy=cam_dy, cam_dz=cam_dz)
        q = Quaternion()
        t = np.array([0,0,0])
        robot = GroundRobot(calib, q, t)

        # Check for correct init
        assert np.all(robot.get_rotation_matrix() == np.eye(3))
        assert np.all(robot.get_extrinsics_matrix() == np.eye(4))

    def test_world_to_robot(self):
        robot = GroundRobot({}, Quaternion(), rob_t=np.array([3, 2, 1]))
        p_world = np.array([[5, 4, 3, 1],
                            [16, 13, 5, 1],
                            [-5, -3, 2, 1]])
        p_rob_emp = robot.transform_points(p_world, Frames.WORLD_FRAME, Frames.BODY_FRAME).T
        p_rob_act = p_world - np.array([3, 2, 1, 0]).reshape(1, 4)
        assert p_rob_act == pytest.approx(p_rob_emp)
        p_world_emp = robot.transform_points(p_rob_emp, Frames.BODY_FRAME, Frames.WORLD_FRAME).T
        assert p_world == pytest.approx(p_world_emp)
        
    def test_points_translation_simple(self):
        calib = get_calib(img_size=IMG_SIZE, cam_dx=cam_dx, cam_dy=cam_dy, cam_dz=cam_dz)
        q = Quaternion()
        t = np.array([5, 2, 1])
        robot = GroundRobot(calib, q, t)

        # 3D points in the camera coordinate frame 
        p_camera = np.array([[0, 0, 1., 1.],
                             [0.5, 0, 1.5, 1.]])
        
        p_world_ref = np.array([[5., 2., 2., 1.], 
                                [5.5, 2., 2.5, 1.]]).T

        # Use the robot frame to convert these to world points
        # Camera frame and body frame are the same
        p_world1 = robot.transform_points(p_camera, Frames.CAM_FRAME, Frames.WORLD_FRAME)
        p_world2 = robot.transform_points(p_camera, Frames.BODY_FRAME, Frames.WORLD_FRAME)

        assert p_world1 == pytest.approx(p_world_ref, 1e-3)
        assert p_world2 == pytest.approx(p_world_ref, 1e-3)

    def test_points_rotation_simple_offset_camera(self):
        # 3D points in the camera coordinate frame 
        p_camera = np.array([[0, 0, 1., 1.],
                            [0.5, 0, 1.5, 1.]])
        
        for placement in ["left", "right", "middle"]:
            # Calibrate the system
            calib = get_calib(img_size=IMG_SIZE, cam_dx=cam_dx, cam_dy=cam_dy, cam_dz=cam_dz, f=1, cam_placement=placement)
            # y is the vector pointing down into the floor
            # q is R^cam_world
            q = Quaternion(axis=[0, 1, 0], angle = -0.7854) # about 45 degrees
            t = np.array([0,0,0])
            robot = GroundRobot(calib, q, t)

            # Hand calcualted reference points
            x1 = np.sqrt(2)/2
            x2 = np.sqrt(2)
            z1 = np.sqrt(2)/2 
            z2 = 3*np.sqrt(2)/4 - np.sqrt(2)/4
            if placement == "left" or placement == "right":
                # Same change in depth
                z1 += cam_dz
                z2 += cam_dz

                # calibration in the x direction is diferent
                # NOTE: Somebody asked me why this is the case.
                # Stick with this intuition, your camera is the only thing moving.
                # In the world, your camera is to the left so your world point is to the left
                if placement == "left":
                    x1 -= cam_dx
                    x2 -= cam_dx
                elif placement == "right":
                    x1 += cam_dx
                    x2 += cam_dx

            # World reference points depending on calibration.
            p_world_ref = np.array([[x1, cam_dy, z1, 1.],
                                    [x2, cam_dy, z2, 1.]]).T

            # Use the robot frame to convert these to world points.
            p_world = robot.transform_points(p_camera, Frames.CAM_FRAME, Frames.WORLD_FRAME)
            assert p_world == pytest.approx(p_world_ref, 1e-3)
 
    def test_project_2d_to_3d_simple(self):
        scale_factor = 1000
        calib = get_calib(img_size=IMG_SIZE, cam_dx=cam_dx, cam_dy=cam_dy, cam_dz=cam_dz, f = scale_factor)
        q = Quaternion()
        t = np.array([0, 0, 0])
        robot = GroundRobot(calib, q, t)
        pix_locs = np.array([[0, 0, 1],
                             [(IMG_SIZE[1]-1)/2, (IMG_SIZE[0]-1)/2, 1],
                             [IMG_SIZE[1]-1, IMG_SIZE[0]-1, 1]])
        depths = np.array([3., 5., 2.]).reshape(3, 1) # meters
        points = pix_locs * depths
        
        world_points = robot.transform_points(points, Frames.IMG_FRAME, Frames.WORLD_FRAME)
        body_points  = robot.transform_points(points, Frames.IMG_FRAME, Frames.BODY_FRAME)
        cam_points   = robot.transform_points(points, Frames.IMG_FRAME, Frames.CAM_FRAME)

        # Hand calculate the reference points
        dist_x = (IMG_SIZE[1]-1)/2
        dist_y = (IMG_SIZE[0]-1)/2
        depths = depths.flatten()
        ref_world_points = np.array([[-dist_x * (depths[0]/scale_factor), -dist_y * (depths[0]/scale_factor), depths[0], 1.],
                                     [0, 0, depths[1], 1.],
                                     [dist_x * (depths[2]/scale_factor), dist_y * (depths[2]/scale_factor), depths[2], 1.]]).T
        assert pytest.approx(world_points, 1e-3) == ref_world_points
        assert pytest.approx(body_points , 1e-3) == ref_world_points
        assert pytest.approx(cam_points  , 1e-3) == ref_world_points

    def test_2d_to_3d_trans(self):
        scale_factor = 1e3
        calib = get_calib(img_size=IMG_SIZE, cam_dx=cam_dx, cam_dy=cam_dy, cam_dz=cam_dz, f=scale_factor)
        q1 = Quaternion(w=0, x=0, y=0, z=1.) # IMU has -x, -y, z basis vectors
        robot = GroundRobot(calib, q1, np.array([3, 5, 4]))
        pix_locs = np.array([[0, 0, 1],
                             [(IMG_SIZE[1]-1)/2, (IMG_SIZE[0]-1)/2, 1],
                             [IMG_SIZE[1]-1, IMG_SIZE[0]-1, 1]])
        depths = np.array([3., 5., 2.]).reshape(3, 1) # meters
        points = pix_locs * depths
        body_points         = robot.transform_points(points, Frames.IMG_FRAME, Frames.BODY_FRAME)
        body_points_w_align = robot.transform_points(points, Frames.IMG_FRAME, Frames.BODY_FRAME_WORLD_ALIGNED)

        # Calculate ref points for body frame
        dist_x = (IMG_SIZE[1]-1)/2
        dist_y = (IMG_SIZE[0]-1)/2
        depths = depths.flatten()
        ref_body_wa_points = np.array([[dist_x * (depths[0]/scale_factor), dist_y * (depths[0]/scale_factor), depths[0], 1.],
                                     [0, 0, depths[1], 1.],
                                     [-dist_x * (depths[2]/scale_factor), -dist_y * (depths[2]/scale_factor), depths[2], 1.]]).T
        
        # See if points make sense
        assert pytest.approx(body_points[2].flatten()) == depths.flatten()
        assert pytest.approx(body_points_w_align[2].flatten()) == depths.flatten()
        assert pytest.approx(ref_body_wa_points * np.array([-1, -1, 1, 1]).reshape(4,1)) == body_points
        assert pytest.approx(body_points_w_align) == ref_body_wa_points

    def test_transform_points_world_aligned(self):
        scale_factor = 1000
        calib = get_calib(img_size=IMG_SIZE, cam_dx=cam_dx, cam_dy=cam_dy, cam_dz=cam_dz, f = scale_factor)
        q1 = Quaternion(w=0, x=0, y=0, z=1.) # IMU has -x, -y, z basis vectors
        t = np.array([5, 10, 15])
        robot = GroundRobot(calib, q1, t)
        world_points= np.array([[5.5, 10.5, 15.5, 1],
                             [4, 9, 14, 1],
                             [3, 15, 10, 1]])

        # Transform to the correct frame
        body_points_aligned = robot.transform_points(world_points, Frames.WORLD_FRAME, Frames.BODY_FRAME_WORLD_ALIGNED)
        body_points         = robot.transform_points(world_points, Frames.WORLD_FRAME, Frames.BODY_FRAME)

        # Test outputs.
        assert world_points[:, :3].T == pytest.approx(body_points_aligned[:3, :] + t.reshape(3, 1))
        assert body_points[:3, :]  == pytest.approx(q1.rotation_matrix @  body_points_aligned[0:3, :])

    # def test_P_simple(self):
    #     f = 1000
    #     # World frame
    #     world_pts = np.array([
    #         [1., 2., 3., 1.],
    #         [4., 5., 6., 1.],
    #         [7., 8., 9., 1.]
    #     ])

    #     # Camera frame
    #     cam_pts = np.array([
    #         [0, -2, -2],
    #         [3, -5, 1],
    #         [6, -6, 4]
    #     ])

    #     # Rotation from world to camera will do this:
    #     # X -> Z
    #     # Y -> X
    #     # Z -> -Y
    #     R_w_to_c = np.array([
    #         [0, 0, 1],
    #         [1, 0, 0],
    #         [0, -1, 0]
    #     ]).T

    #     # from scipy.spatial import transform
    #     # quat = transform.Rotation.from_matrix(R_w_to_c).as_quat()
    #     # rot_mtx = transform.Rotation.from_quat(quat).as_matrix()
    #     # f = Quaternion(quat[-1], quat[0], quat[1], quat[2]).rotation_matrix

    #     # Camera calibration, camera intrinsics and extrinsics
    #     calib = get_calib(img_size=IMG_SIZE, cam_dx=1, cam_dy=2, cam_dz=3, q_c=q_c, f = f)

    #     ground_rob = GroundRobot(calib, q_c, world_t)
        
    #     E = ground_rob.get_E()
    #     f = 0
    #     #cam_pts = P @ world_pts.T

    def test_nuscenes_change_of_basis(self):
        """
        Test a system with nuscenes world frame, robot frame, and camera frame.
        """
        # Change in coordinate system
        # From new to old for column vectors
        # Same as taking old to new and transposing

        T_rob_to_cam = COB.NED_TO_CAM
        
        # Target Locations in the Robot Frame:
        # X, Y, Z, W (DY), H (DZ)
        target_locs = np.array([[2, 3, 0, 0.25, 0.25, 0.25],
                                [1, -3.5, 0.25, 0.25, 0.25, 0.25],
                                [2.5, 1., 0.5, 0.25, 0.25, 0.5]])
        tgt_rot = [Quaternion(),
        Quaternion(),
        Quaternion()]

        # Camera 1 transformation matrix
        cam1_loc = np.array([0., 1.5, 0.])
        cam1_rot = Quaternion(angle = 0, axis = [0, 0, 1])
        T_cam_1  = np.eye(4)
        T_cam_1[0:3, 0:3] = cam1_rot.rotation_matrix
        T_cam_1[0:3, -1]  = -cam1_rot.rotation_matrix@cam1_loc

        # Camera 2 transformation matrix
        cam2_loc = np.array([0., -1.5, 0.])
        cam2_rot = Quaternion(angle = 0, axis = [0, 0, 1])
        T_cam_2  = np.eye(4)
        T_cam_2[0:3, 0:3] = cam2_rot.rotation_matrix
        T_cam_2[0:3, -1]  = -cam2_rot.rotation_matrix@cam2_loc
        
        # Create robot frame with two affine cameras
        calib = {}
        calib["0"] = {}
        calib["1"] = {}
        calib["0"]["K"] = np.array([[200, 0, 256],
                                   [0, 200, 256],
                                   [0, 0, 1]])
        calib["0"]["E"] = T_cam_1
        calib["1"]["K"] = np.array([[200, 0, 256],
                                   [0, 200, 256],
                                   [0, 0, 1]])
        calib["1"]["E"] = T_cam_2

        # Create robot frame, forward 1 meter in the world frame
        rob_frame = GroundRobot(calib, Quaternion(), rob_t = np.array([1, 0, 0]), cob=COB.NED_TO_CAM)

        # 1. Just check if the locations are correct in the camera frame
        # Create the points for the vertiices of the bounding box

        # Check against hand calculated points
        hand_calc_pts_0 = np.array([
                [1.5, 0, 1],
                [-5, 0.25, 0],
                [-0.5, 0.5, 1.5]])
        hand_calc_pts_1 = np.array([
                [4.5, 0, 1],
                [-2, .25, 0],
                [2.5, .5, 1.5]])
        for idx, loc in enumerate(target_locs):
            
            # Homogenous locations for centroids
            loc_homog = np.append(loc[:3], 1).reshape(4, 1)

            # Transform with the class
            cam_pts_0 = rob_frame.transform_points(loc_homog, Frames.WORLD_FRAME, Frames.CAM_FRAME, camera="0")
            cam_pts_1 = rob_frame.transform_points(loc_homog, Frames.WORLD_FRAME, Frames.CAM_FRAME, camera="1")

            assert hand_calc_pts_0[idx] == pytest.approx(cam_pts_0[:3].reshape(3,))
            assert hand_calc_pts_1[idx] == pytest.approx(cam_pts_1[:3].reshape(3,))

    # Transform these points to the camera frame
    # bbox_homog = np.hstack((bbox, np.ones((bbox.shape[0], 1))))