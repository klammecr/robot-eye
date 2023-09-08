# Third Party
import pytest
import numpy as np

# In House
from roboteye.geom import scroll_3d_volume, pi, inv_pi

# CONSTANTS
GRID_RES = (200, 8, 200)

# Fixtures
@pytest.fixture
def calib():
    # Camera intrinsics
    K_met_to_pix = \
        np.array([[10, 0,  127.5],
        [0,  10, 127.5],
        [0,  0,   1]])

    # Camera extrinsics, calibration to the camera, this should be negative 
    # of the translation/rotation of the camera to put in the extrinsics
    # the reason is because the points are static but the camera is moving.
    cam_loc = np.array([1.5, -0.5])
    E_to_cam            = np.eye(4)
    E_to_cam[0:2, -1]   = -cam_loc
    E_from_cam          = np.eye(4)
    E_from_cam[0:2, -1] = cam_loc

    return locals()

class TestProj:
    def test_pi(self, calib):
        """
        This will test projection of 3D points on to an image plane.
        """
        # Make some random 3D points
        pts_w = np.array([[50, 30, 10, 1],
                          [3., 10.25, 36.8, 1.],
                          [1.5, 3.5, 4., 1.]])
        
        # Empirical calculation via function
        emp_pts, mask = pi(calib["K_met_to_pix"], calib["E_to_cam"], pts_w.T, 256, 256)

        # Actual hand-calculation
        actual_pts = []
        for pt in pts_w:
            # 1. Offset by extrinsics to camera
            # 2. Scale by intrinsic ratio to convert to pixels
            # 3. Offset by principal point to get pixel location
            depth = (pt[2] + calib["E_to_cam"][2, -1]) * calib["K_met_to_pix"][2, 2]
            v     = (pt[0] + calib["E_to_cam"][0, -1]) * calib["K_met_to_pix"][0, 0] + calib["K_met_to_pix"][0, -1] * pt[2]
            u     = (pt[1] + calib["E_to_cam"][1, -1]) * calib["K_met_to_pix"][1, 1] + calib["K_met_to_pix"][1, -1] * pt[2]

            # Divide by depth to turn into 2D homogenous coordinates
            v /= depth
            u /= depth
            
            # Add to the list of points
            actual_pts.append([v, u, depth])

        # Test empirical vs. actual
        actual_pts = np.array(actual_pts).T
        assert actual_pts == pytest.approx(emp_pts)
        assert np.all(mask)

    def test_inv_pi(self, calib):
        """
        This will test unprojecting 2D points on an image to 3D points.
        """
        img_w  = 256
        img_h  = 512
        uv_pts = np.array([[100, 200],
                           [150, 150],
                           [0, 0]])
        depths=np.array([3., 1.5, 6.2])
        
        # Project to 3D
        # NOTE: I say world here but to the world and the robot are aligned and located at the same place
        emp_pts_w = inv_pi(np.linalg.inv(calib["K_met_to_pix"]), calib["E_from_cam"], uv_pts, depths=depths)

        K_to_met = np.linalg.inv(calib["K_met_to_pix"])
        pts_w = []
        for i, pt in enumerate(uv_pts):
            v, u = pt
            x = (depths[i] * v * K_to_met[0, 0] + K_to_met[0, 2] * depths[i]) + calib["E_from_cam"][0, -1]
            y = (depths[i] * u * K_to_met[1, 1] + K_to_met[1, 2] * depths[i]) + calib["E_from_cam"][1, -1]
            z = (depths[i] * K_to_met[2, 2]) + calib["E_from_cam"][2, -1]
            pts_w.append([x, y, z])

        assert np.array(pts_w) == pytest.approx(emp_pts_w[:-1].T)


    def test_pi_then_inv_pi(self, calib):
        """
        Validate that projecting then unprojecting gives back the same points.
        """
        # Make some random 3D points
        pts_w = np.array([[50, 30, 10, 1],
                          [3., 10.25, 36.8, 1.],
                          [1.5, 3.5, 4., 1.]])
        
        # Empirical calculation via function
        emp_pts, mask = pi(calib["K_met_to_pix"], calib["E_to_cam"], pts_w.T, 256, 256)
        locs   = emp_pts[:-1]
        depths = emp_pts[-1]
        emp_pts_w = inv_pi(np.linalg.inv(calib["K_met_to_pix"]), calib["E_from_cam"], locs, depths)

        assert pts_w == pytest.approx(emp_pts_w.T)

    def test_inv_pi_then_pi(self):
        """
        Validating that unprojecting then reprojecting give back the same points.
        """
        pass

class TestScroll3DVolume:
    def test_int_scroll(self):
        """
        This test will do scrolling for a scalar (3D volume).
        Here there is no translation.
        """  
        volume = np.zeros((200, 8, 200))
        volume[100, 5, 100] = 3.
        volume[101, 5, 100] = 3.
        volume[102, 5, 100] = 3.
        volume[103, 5, 100] = 3.
        new_vol = scroll_3d_volume(GRID_RES, [volume], [0, 0, 0])[0]
        assert new_vol.shape == volume.shape
        assert new_vol[100, 5, 100] == pytest.approx(3., 1e-2)
        assert new_vol[101, 5, 100] == pytest.approx(3., 1e-2)
        assert new_vol[102, 5, 100] == pytest.approx(3., 1e-2)
        assert new_vol[103, 5, 100] == pytest.approx(3., 1e-2)

    def test_int_scroll_translation(self):
        """
        This test will do scrolling for a scalar (3D volume).
        Here there is translation that is an integer.
        """  
        volume = np.zeros((200, 8, 200))
        t      = np.array([1., 2., 3.])
        volume[0, 0, 0]     = 15.
        volume[100, 5, 100] = 3.
        volume[101, 5, 100] = 3.
        volume[102, 5, 100] = 3.
        volume[103, 5, 100] = 3.
        new_vol = scroll_3d_volume(GRID_RES, [volume], t)[0]
        assert new_vol.shape == volume.shape
        assert new_vol[100, 5, 100] == pytest.approx(0.)
        assert new_vol[101, 5, 100] == pytest.approx(0.)
        assert new_vol[102, 5, 100] == pytest.approx(0.)
        assert new_vol[103, 5, 100] == pytest.approx(0.)
        assert new_vol[0, 0, 0]     == pytest.approx(0., 1e-2)
        assert new_vol[99,  int(5 - t[1]), int(100 - t[2])] == pytest.approx(3., 1e-2)
        assert new_vol[100, int(5 - t[1]), int(100 - t[2])] == pytest.approx(3., 1e-2)
        assert new_vol[101, int(5 - t[1]), int(100 - t[2])] == pytest.approx(3., 1e-2)
        assert new_vol[102, int(5 - t[1]), int(100 - t[2])] == pytest.approx(3., 1e-2)
        
    def test_float_scroll(self):
        volume = np.zeros((200, 8, 200))
        t      = np.array([1.5, 2.5, 3.5])
        volume[100, 5, :] = 3.
        volume[101, 5, :] = 4.
        volume[102, 5, :] = 5.
        volume[103, 5, :] = 6.
        new_vol = scroll_3d_volume(GRID_RES, [volume], t, order=1)[0]

        # Check the interpolation for order 1 (linear)
        assert new_vol[98, 2, 96]  == pytest.approx(3/4, 1e-2)
        assert new_vol[99, 2, 96]  == pytest.approx(7/4, 1e-2)
        assert new_vol[100, 2, 96] == pytest.approx(9/4, 1e-2)
        assert new_vol[101, 2, 96] == pytest.approx(11/4, 1e-2)
        assert new_vol[102, 2, 96] == pytest.approx(6/4, 1e-2)

    def test_int_scroll_4d(self):
        """
        This will test the functionality for vectors at 3D locations (4D)
        """
        # Create helper arrays
        volume    = np.zeros((200, 8, 200, 64))
        vol_shape = volume.shape
        t         = np.array([1, 2, 3])

        # Set the vectors in the volume locs
        volume[0, 0, 0]    = np.ones(64) * 26
        volume[50, 3, 50]  = np.ones(64)
        volume[-1, -1, -1] = np.ones(64) * 50
        new_vol = scroll_3d_volume(GRID_RES, [volume], t, order=1)[0]

        # Scrolled past the all 26s
        assert np.all([not match_arr_dim for match_arr_dim in np.where(new_vol == 26)])
        # Still in frame
        assert np.all(new_vol[50 - t[0], 3 - t[1], 50 - t[2]] == 1)
        # The last voxel is now somewhere in the middleish
        last_loc = (vol_shape[0]-(t[0]+1), vol_shape[1]-(t[1]+1), vol_shape[2]-(t[2]+1))
        assert np.all(new_vol[last_loc] == 50)

    def test_float_scroll_4d(self):
        """
        This test will do the scrolling for a vector (4D volume).
        This should test some simple interpolation just for a sanity check.
        """
        # Create helper arrays
        volume    = np.zeros((200, 8, 200, 64))
        vol_shape = volume.shape
        t         = np.array([1.5, 2.5, 3.5])

        # Set the vectors in the volume locs
        volume[49, 3, :] = np.ones(64) * 26
        volume[50, 3, :] = np.ones(64)
        volume[51, 3, :] = np.ones(64) * 50
        new_vol = scroll_3d_volume(GRID_RES, [volume], t, order=1)[0]

        # Test the interpolation
        assert np.all(new_vol[47, 0, 100] == 26/4)
        assert np.all(new_vol[48, 0, 100] == 27/4)
        assert np.all(new_vol[49, 0, 100] == 51/4)
        assert np.all(new_vol[50, 0, 100] == 50/4)

