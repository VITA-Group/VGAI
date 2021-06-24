import sys
from os.path import abspath, dirname, join
import airsim
import numpy as np
import os
import tempfile
import pprint
from time import sleep
import random
import re
import torch
import pickle
import time
from problems.flocking import FlockingProblem
import cv2
import threading

# N - number of drones
# dist - dist between drones on circumference, 0.5 < 0.75 keeps things interesting
def circle_helper(N, dist):
    r = dist * N / 2 / np.pi
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).reshape((N, 1))
    # angles2 = np.pi - angles
    return r * np.hstack((np.cos(angles), np.sin(angles))), -0.5 * np.hstack((np.cos(angles), -0.5 * np.sin(angles)))


def circle(N):
    if N <= 20:
        return circle_helper(N, 0.5)
    else:
        smalln = int(N * 2.0 / 5.0)
        circle1, v1 = circle_helper(smalln, 0.5)
        circle2, v2 = circle_helper(N - smalln, 0.5)
        return np.vstack((circle1, circle2)), np.vstack((v1, v2))


def grid(N):
    side = 5
    side2 = int(N / side)
    xs = np.arange(0, side) - side / 2.0
    ys = np.arange(0, side2) - side2 / 2.0
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.reshape((N, 1))
    ys = ys.reshape((N, 1))
    return 0.6 * np.hstack((xs, ys))


def twoflocks(N):
    half_n = int(N / 2)
    grid1 = grid(half_n)
    delta = 6
    grid2 = grid1.copy() + np.array([0, delta / 2]).reshape((1, 2))
    grid1 = grid1 + np.array([0, -delta / 2]).reshape((1, 2))

    vels1 = np.tile(np.array([-1.0, delta]).reshape((1, 2)), (half_n, 1))
    vels2 = np.tile(np.array([1.0, -delta]).reshape((1, 2)), (half_n, 1))

    grids = np.vstack((grid1, grid2))
    velss = 0.05 * np.vstack((vels1, vels2))

    return grids, velss


def parse_settings(fname):
    names = []
    homes = []
    for line in open(fname):
        for n in re.findall(r'\"(.+?)\": {', line):
            if n != 'Vehicles':
                names.append(n)
        p = re.findall(r'"X": ([-+]?\d*\.*\d+), "Y": ([-+]?\d*\.*\d+), "Z": ([-+]?\d*\.*\d+)', line)
        if p:
            homes.append(np.array([float(p[0][0]), float(p[0][1]), float(p[0][2])]).reshape((1, 3)))
    return names, np.concatenate(homes, axis=0)


def quaternion_to_yaw(q):
    # yaw (z-axis rotation) from quaternion
    w = float(q.w_val)
    x = float(q.x_val)
    y = float(q.y_val)
    z = float(q.z_val)
    siny_cosp = +2.0 * (w * z + x * y)
    cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw



def getStates(N, names, client, home):
    states = np.zeros(shape=(N, 4))
    yaws = np.zeros(shape=(N, 1))
    for i in range(0, N):
        state = client.getMultirotorState(vehicle_name=names[i])
        states[i][0] = float(state.kinematics_estimated.position.x_val) + home[i][0]
        states[i][1] = float(state.kinematics_estimated.position.y_val) + home[i][1]
        states[i][2] = float(state.kinematics_estimated.linear_velocity.x_val)
        states[i][3] = float(state.kinematics_estimated.linear_velocity.y_val)
        yaws[i] = quaternion_to_yaw(state.kinematics_estimated.orientation)

    return states, yaws


def setup_drones(N, names, client):
    for i in range(0, N):
        client.enableApiControl(True, names[i])
    for i in range(0, N):
        client.armDisarm(True, names[i])

    fi = []
    for i in range(N):
        fi.append(client.takeoffAsync(vehicle_name=names[i]))  # .join()
    for f in fi:
        f.join()


def send_velocity_commands(client, N, ut1, z, duration, names):
    fi = []
    for i in range(N):
        fi.append(client.moveByVelocityZAsync(ut1[i, 0], ut1[i, 1], z[i], duration, vehicle_name=names[i]))
    sleep(0.1)
    for f in fi:
        f.join()


def send_loc_commands(client, N, x0, home, mean_x, mean_y, z, names):
    fi = []
    for i in range(N):
        fi.append(client.moveToPositionAsync(x0[i][0] - home[i][0] + mean_x, x0[i][1] - home[i][1] + mean_y, z[i], 6.0,
                                             vehicle_name=names[i]))
    sleep(0.1)
    for f in fi:
        f._timeout = 40  # quads sometimes get stuck during a crash and never reach the destination
        f.join()



def send_accel_commands(client, N, u, duration, z, names):
    fi = []
    for i in range(N):
        fi.append(client.moveByAngleZAsync(float(u[i, 0]), float(u[i, 1]), z[i], 0.0, duration, vehicle_name=names[i]))
    for f in fi:
        f.join()


def display_msg(client, msg):
    print(msg)
    client.simPrintLogMessage(msg)



def get_ut(client, N, names, model):
    camera_names = ['front_center', 'front_right', 'front_left', 'back_center']
    camera_idx = ['0', '1', '2', '3']
    
    client.simPause(True)
    ut1 = np.zeros((N, 2))
    for i in range(N):
        #client.simSetCameraOrientation("front_left", airsim.to_quaternion(0,  0,  -0.82 ), vehicle_name=names[i]);
        #client.simSetCameraOrientation("front_right", airsim.to_quaternion(0, 0,   0.82 ), vehicle_name=names[i]);
        responses = client.simGetImages([
                        airsim.ImageRequest('front_center', airsim.ImageType.Scene),
                        airsim.ImageRequest('front_right', airsim.ImageType.Scene),
                        airsim.ImageRequest('front_left', airsim.ImageType.Scene),
                        airsim.ImageRequest('back_center', airsim.ImageType.Scene)], vehicle_name=names[i])
             
        for j, response in enumerate(responses):
        
            if response.pixels_as_float:
                #print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                airsim.write_pfm(os.path.normpath(os.path.join(tmp_img_dir, filename + '.pfm')), airsim.get_pfm_array(response))
            else:
                #print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                airsim.write_file(os.path.normpath(os.path.join('tempfile' + '.png')), response.image_data_uint8)
                if j == 0:
                    f_image = cv2.imread('tempfile.png')
                if j == 1:
                    r_image = cv2.imread('tempfile.png')
                if j == 2:
                    l_image = cv2.imread('tempfile.png')
                if j == 3:
                    b_image = cv2.imread('tempfile.png')

         
        image = np.concatenate((f_image, l_image, r_image, b_image), axis=1)
        image = image.transpose(2,0,1)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float().cuda()
        x, y = model(image)
        ut1[i,0] = x
        ut1[i,1] = y
        
    client.simPause(False)  
    return ut1

def record_agg3(client, N, xt1, ut1, names, exp_idx, time_step, x_agg):
    tmp_img_states = os.path.join('/home/tkhu/Documents/AirSim/airsim_drone_1022_states/')
    tmp_img_actions = os.path.join('/home/tkhu/Documents/AirSim/airsim_drone_1022_actions/')


    tmp_img_aggs = os.path.join('/home/tkhu/Documents/AirSim/airsim_drone_1022_aggs/')

    client.simPause(True)
    sleep(0.01)
    
    
    for i in range(N):
        filename = 'exp' + str(exp_idx) + '_time' + str(time_step) + '_' + names[i]
        np.savetxt(tmp_img_states + filename + '.txt', xt1[i, :], delimiter=' ')
        np.savetxt(tmp_img_actions + filename + '.txt',ut1[i, :], delimiter=' ')
        np.savetxt(tmp_img_aggs + filename + '.txt',x_agg[i, :], delimiter=' ')

        
    client.simPause(False)  




def record_images(client, N, xt1, ut1, names, exp_idx, time_step, x_aggs, x_feats, root_path):
     
    feats_dir = os.path.join(root_path, 'feats') 
    aggs_dir = os.path.join(root_path, 'aggs') 
    imgs_dir = os.path.join(root_path, 'imgs') 
    states_dir = os.path.join(root_path, 'states') 
    actions_dir = os.path.join(root_path, 'actions') 

    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)    
    if not os.path.exists(aggs_dir):
        os.makedirs(aggs_dir)    
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)    
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)    
    if not os.path.exists(actions_dir):
        os.makedirs(actions_dir)    
     
    camera_names = ['front_center', 'front_right', 'front_left', 'back_center']
    camera_idx = ['0', '1', '2', '3']
    
    client.simPause(True)
    
    
    for i in range(N):
        #client.simSetCameraOrientation('2', airsim.to_quaternion(0,  0,  -0.82 ), vehicle_name=names[i])
        #client.simSetCameraOrientation('1', airsim.to_quaternion(0, 0,   0.82 ), vehicle_name=names[i])
        responses = client.simGetImages([
                        airsim.ImageRequest('front_center', airsim.ImageType.Scene),
                        airsim.ImageRequest('front_right', airsim.ImageType.Scene),
                        airsim.ImageRequest('front_left', airsim.ImageType.Scene),
                        airsim.ImageRequest('back_center', airsim.ImageType.Scene)], vehicle_name=names[i])
             
        for j, response in enumerate(responses):
        
            filename = 'exp' + str(exp_idx) + '_time' + str(time_step) + '_' + names[i]+'_'+ camera_idx[j]
            if response.pixels_as_float:
                #print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                airsim.write_pfm(os.path.normpath(os.path.join(imgs_dir, filename + '.pfm')), airsim.get_pfm_array(response))
            else:
                #print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                airsim.write_file(os.path.normpath(os.path.join(imgs_dir, filename + '.png')), response.image_data_uint8)

        img_filename = filename
        image = cv2.imread(os.path.normpath(os.path.join(imgs_dir, img_filename + '.png')))
        if type(image) == type(None):
            client.simPause(False)  
            return False



        filename = 'exp' + str(exp_idx) + '_time' + str(time_step) + '_' + names[i]
        np.savetxt(os.path.join(states_dir, filename+'.txt'), xt1[i,:],delimiter=' ')
        np.savetxt(os.path.join(actions_dir, filename+'.txt'), ut1[i,:],delimiter=' ')
        np.savetxt(os.path.join(feats_dir, filename+'.txt'), x_feats[i,:],delimiter=' ')
        np.savetxt(os.path.join(aggs_dir, filename+'.txt'), x_aggs[i,:],delimiter=' ')
    #print('image path = {}'.format(os.path.normpath(os.path.join(imgs_dir, img_filename + '.png'))))
    client.simPause(False)  
    return True
    


def get_imgs(client, N, names):
    client.simPause(True)
    camera_names = ['front_center', 'front_right', 'front_left', 'back_center']
    camera_idx = ['0', '1', '2', '3']
    image = []
    for i in range(N):
        #client.simSetCameraOrientation("front_left", airsim.to_quaternion(0,  0,  -0.82 ), vehicle_name=names[i]);
        #client.simSetCameraOrientation("front_right", airsim.to_quaternion(0, 0,   0.82 ), vehicle_name=names[i]);
        responses = client.simGetImages([
                        airsim.ImageRequest('front_center', airsim.ImageType.Scene),
                        airsim.ImageRequest('front_right', airsim.ImageType.Scene),
                        airsim.ImageRequest('front_left', airsim.ImageType.Scene),
                        airsim.ImageRequest('back_center', airsim.ImageType.Scene)], vehicle_name=names[i])
        fovs = []     
        for j, response in enumerate(responses):
        
            if response.pixels_as_float:
                airsim.write_pfm(os.path.normpath(os.path.join(tmp_img_dir, filename + '.pfm')), airsim.get_pfm_array(response))
            else:
                airsim.write_file(os.path.normpath(os.path.join('tempfile' + '.png')), response.image_data_uint8)
                if j == 0:
                    f_image = cv2.imread('tempfile.png')
                if j == 1:
                    r_image = cv2.imread('tempfile.png')
                if j == 2:
                    l_image = cv2.imread('tempfile.png')
                if j == 3:
                    b_image = cv2.imread('tempfile.png')
        if type(f_image) == type(None):

            client.simPause(False)  
            return None

        if type(r_image) == type(None):
            return None

        if type(l_image) == type(None):
            return None

        if type(b_image) == type(None):
            return None

        fov = np.concatenate([f_image, l_image, r_image, b_image], axis=1)
        fov = fov.swapaxes(0, 2).swapaxes(1, 2)
        fovs.append(np.expand_dims(fov, axis=0)) # 1 x 3 x 144 x 1024
        image.append(fovs)
    image = np.concatenate(image, axis=0) # 50x3x144x1024
    client.simPause(False)  
    return np.squeeze(image, axis=1)
