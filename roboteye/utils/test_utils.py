# Third Party
import numpy as np
from pyquaternion import Quaternion

def create_K(f, img_size):
    return np.array([[f, 0, (img_size[1]-1)/2],
                     [0, f, (img_size[0]-1)/2],
                     [0, 0, 1]])

def create_extrinsics(cam_dx, cam_dy, cam_dz, cam_placement):
    E = np.eye(4)   
    if cam_placement == "left":
        E[:3, -1] = np.array([-cam_dx, 0, cam_dz])
    elif cam_placement == "right":
        E[:3, -1] = np.array([cam_dx, 0, cam_dz])
    # This should be the inverse of the camera locations to transform pts
    return np.linalg.inv(E)

def get_calib(img_size, cam_dx, cam_dy, cam_dz, q_c = Quaternion(1, 0, 0, 0), f = 1, cam_placement = "middle"):
    """
    Create camera calibration for testing purposes.

    \param[in] img_size:      (H,W) of size of image plane (pix)
    \param[in] q_c:           Quaternion describing the rotation of the camera (including change of basis)
    \param[in] cam_dx:        X displacement of camera from world origin (m)
    \param[in] cam_dy:        Y displacement of camera from world origin (m)
    \param[in] cam_dz:        Z displacement of camera from world origin (m)
    \param[in] f:             Focal length used for scaling meters to pixels
    \param[in] cam_placement: Where the camera is relative to the ego frame

    \return: Calibration dictionary for a robot with as single camera.
    """
    calib = {}
    calib["K"] = create_K(f, img_size)
    cam_rot = np.eye(4)
    cam_rot[:3, :3] = q_c.rotation_matrix
    calib["E"] =  cam_rot @ create_extrinsics(cam_dx, cam_dy, cam_dz, cam_placement)
    return calib

def get_calib_K(K, cam_dx, cam_dy, cam_dz, cam_placement="middle"):
    calib     = {}
    calib["K"] = K
    calib["E"] = create_extrinsics(cam_dx, cam_dy, cam_dz)
    return calib

def setup_camera_fov(hfov, wid, hgt, sens_width=13.2):
    """
    Setup a camera that is able to see a certain number of meters up down left and right.

    \param[in] hfov: Horizontal field of view (degrees)
    \param[in] wd: Working distance (mm)
    \param[in] sens_width: Width of the sensor (mm)
    """
    hfov_rad = hfov * np.pi/180
    f = sens_width / (2 * np.tan(hfov_rad/2))
    K = np.array([[f, 0, (wid-1/2)],
                  [0, f, (hgt-1)/2],
                  [0, 0, 1]])
    return K

def create_bbox_pts(obj_bboxes, obj_rots):
    final_pts = []
    for obj_num in range(obj_bboxes.shape[0]):
        # Compute the verticies then rotate them
        obj_bbox = obj_bboxes[obj_num]
        obj_rot  = obj_rots[obj_num]
        xc, yc, zc, dx, dy, dz = obj_bbox
        bounding_pts = []
        for x_off in [-dx/2, dx/2]:
            for y_off in [-dy/2, dy/2]:
                for z_off in [-dz/2, dz/2]:
                    # Just rotate the offset because this is rotating relative to the
                    # centroid of the robot
                    offset = obj_rot.rotate([x_off, y_off, z_off])
                    bounding_pts.append(np.array([xc, yc, zc]) + np.array(offset))
        final_pts.append(bounding_pts)
    return np.array(final_pts)