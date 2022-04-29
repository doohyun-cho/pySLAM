"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

from pyexpat.model import XML_CTYPE_CHOICE
import numpy as np
np.random.seed(1)
import cv2
import math
import time 

import platform 

from config import Config

from slam import Slam, SlamState
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory

#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

if platform.system()  == 'Linux':     
    from display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!

from viewer3D import Viewer3D
from utils_sys import getchar, Printer 

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from feature_tracker_configs import FeatureTrackerConfigs

from parameters import Parameters  
import multiprocessing as mp 
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from utils_geom import add_ones_1D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4)


if __name__ == "__main__":

    config = Config()

    dataset = dataset_factory(config.dataset_settings)

    groundtruth = groundtruth_factory(config.dataset_settings)
    # groundtruth = None # not actually used by Slam() class; could be used for evaluating performances 

    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])
    
    num_features=2000 

    # tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn 
    tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching 

    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT, CONTEXTDESC
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config['num_features'] = num_features
    tracker_config['tracker_type'] = tracker_type
    
    print('tracker_config: ',tracker_config)    
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    # create SLAM object 
    slam = Slam(cam, feature_tracker, groundtruth)
    time.sleep(1) # to show initial messages 

    viewer3D = Viewer3D()
    
    if platform.system()  == 'Linux':    
        display2d = Display2D(cam.width, cam.height)  # pygame interface 
    else: 
        display2d = None  # enable this if you want to use opencv window

    matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')    

    do_step = False   
    is_paused = False 
    
    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed 
    fig, axs = plt.subplots(ncols=2, figsize=(13,7), subplot_kw={"projection":"3d"})

    thr_dist = 5
    height_min, height_max = 1.3, 2.2   # I don't know the relative scale...                    
    width_max = 2
    forward_max = 10
    width_min, forward_min = -width_max, -forward_max
    gap = 0.4
    xx = np.arange(width_min, width_max+.01, gap)
    zz = np.arange(forward_min/2, forward_max/2+.01, gap)
    xx, zz = np.meshgrid(xx, zz)
    xx, zz = np.ravel(xx), np.ravel(zz)
    X_pred = np.vstack((xx, zz)).T
    ub = forward_max
    y_scaler = StandardScaler()

    while dataset.isOk():
            
        if not is_paused: 
            print('..................................')
            print('image: ', img_id)                
            img = dataset.getImageColor(img_id)
            if img is None:
                print('image is empty')
                getchar()
            timestamp = dataset.getTimestamp()          # get current timestamp 
            next_timestamp = dataset.getNextTimestamp() # get next timestamp 
            frame_duration = next_timestamp-timestamp 

            if img is not None:
                time_start = time.time()                  
                slam.track(img, img_id, timestamp)  # main SLAM function 
                                
                # 3D display (map display)
                if viewer3D is not None:
                    viewer3D.draw_map(slam)
                
                map_points = slam.map.get_points()
                if map_points:
                    # current cam pose
                    frame_ = slam.map.get_frame(-1)
                    Twc = frame_.Twc.copy()
                    Tcw = frame_.Tcw.copy()
                    Rwc, Rcw = Twc[:3,:3], Tcw[:3,:3]
                    twc, tcw = Twc[:3,3], Tcw[:3,3]

                    # get current heading vector by multiplying R at init heading, R@v
                    v_heading = Rwc @ np.array([0,0,1]).reshape(-1,1)

                    r = Rotation.from_matrix(Rwc)
                    print('cur pos: ', twc)
                    print('Euler (deg): ', r.as_euler('xyz', degrees=True))
                    print('v_heading: ', v_heading.ravel())

                    pts_near_cam = []
                    pts_c = [(Tcw@np.hstack((p.pt, 1)))[:3] for p in map_points]
                    # pts = [(Rcw@p.pt.reshape(-1,1)).T for p in map_points]  # change from world to cam coordinate
                    for pt in pts_c:
                        if (width_min < pt[0] < width_max) \
                            and (height_min < pt[1] < height_max) \
                            and (forward_min < pt[2] < forward_max):
                            pts_near_cam.append(pt)

                    if pts_near_cam:
                        pts_near_cam = np.array(pts_near_cam)
                        fontlabel = {"fontsize":"large", "color":"black", "fontweight":"bold"}
                        ax = axs[0]
                        ax.clear()
                        Xs, Ys, Zs = pts_near_cam[:,0], pts_near_cam[:,1], pts_near_cam[:,2]
                        ax.scatter(Xs, Zs, Ys)
                        ax.set_xlabel("X", fontdict=fontlabel)
                        ax.set_ylabel("Z", fontdict=fontlabel)
                        ax.set_title("cam coord\nY", fontdict=fontlabel)
                        ax.invert_zaxis()
                        
                        ax = axs[1]
                        ax.clear()
                        ax.scatter(Xs, Zs, Ys)
                        ax.set_xlim3d([-ub,ub])
                        ax.set_ylim3d([-ub,ub])
                        ax.set_zlim3d([-ub,ub])
                        ax.set_xlabel("X", fontdict=fontlabel)
                        ax.set_ylabel("Z", fontdict=fontlabel)
                        ax.set_title("cam coord\nY", fontdict=fontlabel)
                        ax.invert_zaxis()
                        # ax.set_box_aspect([1,1,1])  # set equal ratio
                        plt.show()
                        time.sleep(0.01)
                        plt.show()
                        time.sleep(0.01)

                        z_min, z_max = np.min(Zs), np.max(Zs)
                        if (z_min < 0 < z_max) and (z_max - z_min > 5) and pts_near_cam.shape[0] > 20:
                            kernel = 1 * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2))
                            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
                            X_train = pts_near_cam[:,[0,2]]
                            y_train = pts_near_cam[:,1]
                            y_scaled = y_scaler.fit_transform(y_train.reshape(-1,1))
                            gp.fit(X_train, y_scaled)
                            
                            yy_scaled = gp.predict(X_pred)
                            yy = y_scaler.inverse_transform(yy_scaled)

                            # z direction
                            dz = 4  # wheel dist?
                            X_forward = np.array([[0,-dz], [0,0]]) # front wheels at 0, back wheels at -dz
                            y_forward = y_scaler.inverse_transform(gp.predict(X_forward)).ravel()
                            dy = y_forward[1] - y_forward[0]
                            vec_road = np.array([0,dy,dz])
                            vec_road = vec_road / np.linalg.norm(vec_road)
                            pitch_deg = np.arcsin(-vec_road[1]) * 180 / np.pi

                            # x direction
                            dx = 2  # wheel dist?
                            X_right = np.array([[-dx/2,0], [dx/2,0]])
                            y_right = y_scaler.inverse_transform(gp.predict(X_right)).ravel()
                            dy = y_right[1] - y_right[0]
                            vec_road = np.array([dx,dy,0])
                            vec_road = vec_road / np.linalg.norm(vec_road)
                            roll_deg = np.arcsin(vec_road[1]) * 180 / np.pi
                            
                            # height
                            X_height = np.array([[-dx/2,0], [dx/2,0]]) 
                            y_height = y_scaler.inverse_transform(gp.predict(X_height)).ravel()
                            height_cur = np.mean(y_height)

                            axs[0].scatter(xx, zz, yy, alpha=0.4)
                            axs[0].set_title(f"cam coord\nY\n{height_cur:.3f}m, p {pitch_deg:.3f}deg, r {roll_deg:.3f}deg", fontdict=fontlabel)
                            axs[1].scatter(xx, zz, yy, alpha=0.4)

                            # 3dplot - pitch
                            axs[1].plot([0,0], [0,0], [0,height_cur], c='k')
                            axs[1].plot([0,0], [-dz,-dz], [0,height_cur], c='k')

                            # 3dplot - yaw
                            axs[1].plot([-dx/2,-dx/2], [0,0], [0,height_cur], c='b')
                            axs[1].plot([dx/2,dx/2], [0,0], [0,height_cur], c='b')
                            print()

                    # pts_near_cam = [p for p in map_points \
                    #     if (np.linalg.norm(pt - twc) < thr_dist) \
                    #         and (pt[1] - twc[1] < thr_height)]
                    print('# road pts: ', len(pts_near_cam))
                    if Parameters.kUseGroundTruthScale:
                        slam.tracking.get_absolute_scale(frame_.id)
                        print('true x, y, z: ', slam.tracking.trueX, slam.tracking.trueY, slam.tracking.trueZ)
                    print()

                img_draw = slam.map.draw_feature_trails(img)
                    
                # 2D display (image display)
                if display2d is not None:
                    display2d.draw(img_draw)
                else: 
                    cv2.imshow('Camera', img_draw)

                if matched_points_plt is not None: 
                    if slam.tracking.num_matched_kps is not None: 
                        matched_kps_signal = [img_id, slam.tracking.num_matched_kps]     
                        matched_points_plt.draw(matched_kps_signal,'# keypoint matches',color='r')                         
                    if slam.tracking.num_inliers is not None: 
                        inliers_signal = [img_id, slam.tracking.num_inliers]                    
                        matched_points_plt.draw(inliers_signal,'# inliers',color='g')
                    if slam.tracking.num_matched_map_points is not None: 
                        valid_matched_map_points_signal = [img_id, slam.tracking.num_matched_map_points]   # valid matched map points (in current pose optimization)                                       
                        matched_points_plt.draw(valid_matched_map_points_signal,'# matched map pts', color='b')  
                    if slam.tracking.num_kf_ref_tracked_points is not None: 
                        kf_ref_tracked_points_signal = [img_id, slam.tracking.num_kf_ref_tracked_points]                    
                        matched_points_plt.draw(kf_ref_tracked_points_signal,'# $KF_{ref}$  tracked pts',color='c')   
                    if slam.tracking.descriptor_distance_sigma is not None: 
                        descriptor_sigma_signal = [img_id, slam.tracking.descriptor_distance_sigma]                    
                        matched_points_plt.draw(descriptor_sigma_signal,'descriptor distance $\sigma_{th}$',color='k')                                                                 
                    matched_points_plt.refresh()    
                
                # duration = time.time()-time_start 
                # if(frame_duration > duration):
                #     print('sleeping for frame')
                #     time.sleep(frame_duration-duration)        
                    
            img_id += 1  
        else:
            time.sleep(1)                                 
        
        # get keys 
        key = matched_points_plt.get_key()  
        key_cv = cv2.waitKey(1) & 0xFF    
        
        # manage interface infos  
        
        if slam.tracking.state==SlamState.LOST:
            if display2d is not None:     
                getchar()                              
            else: 
                key_cv = cv2.waitKey(0) & 0xFF   # useful when drawing stuff for debugging 
         
        if do_step and img_id > 1:
            # stop at each frame
            if display2d is not None:            
                getchar()  
            else: 
                key_cv = cv2.waitKey(0) & 0xFF         
        
        if key == 'd' or (key_cv == ord('d')):
            do_step = not do_step  
            Printer.green('do step: ', do_step) 
                      
        if key == 'q' or (key_cv == ord('q')):
            if display2d is not None:
                display2d.quit()
            if viewer3D is not None:
                viewer3D.quit()
            if matched_points_plt is not None:
                matched_points_plt.quit()
            break
        
        if viewer3D is not None:
            is_paused = not viewer3D.is_paused()         
                        
    slam.quit()
    
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
