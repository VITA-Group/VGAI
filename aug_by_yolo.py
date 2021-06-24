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
from utils import *
import argparse
from network_agg import Network as Network_Agg
from network_yolo import Network as Network_Feat



parser = argparse.ArgumentParser("trace")
parser.add_argument('--model_feat_path', type=str, default='None', help='path to save the model')
parser.add_argument('--model_agg_path', type=str, default='None', help='path to save the model')
args = parser.parse_args()


def main():
    ## load model
    model_feat = Network_Feat() 
    params = torch.load(args.model_feat_path)['state_dict']
    model_feat.load_state_dict(params)
    model_feat = torch.nn.DataParallel(model_feat).cuda()
    model_feat.eval()

    model_agg = Network_Agg() 
    params = torch.load(args.model_agg_path)['state_dict']
    model_agg.load_state_dict(params)
    model_agg = torch.nn.DataParallel(model_agg).cuda()
    model_agg.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 416
    model_def = '/home/grads/t/tkhu/PyTorch-YOLOv3-tkhu/config/yolov3-custom.cfg'
    weights_path = '/home/grads/t/tkhu/PyTorch-YOLOv3-tkhu/checkpoints/yolov3_ckpt_99.pth'
    model_yolo = Darknet(model_def, img_size=img_size).to(device)
    model_yolo.load_state_dict(torch.load(weights_path))
   
    problem = pickle.load(open('problem.pkl', 'rb'))
    problem.comm_radius = 1.5
    # parse settings file with drone names and home locations
    fname = '/home/grads/t/tkhu/Documents/AirSim/settings.json'
    names, home = parse_settings(fname)
    n_agents = len(names)
    problem.n_nodes = n_agents
    
    #####################
    # use centralized controller
    problem.centralized_controller = True

    # rescale locations and velocities to avoid changing potential function
    scale = 6
    init_scale = 6

    # duration of velocity commands
    duration = 0.01  # duration of each control action used for training
    true_dt = 1.0 / 7.5  # average of actual measurements

    # make the problem easier because of slow control loop frequency
    problem.v_max = 3
    print(problem.v_max)
    problem.v_max = problem.v_max * 0.075
    problem.v_bias = problem.v_bias * 0.3


    

    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    display_msg(client, 'Initializing...')

    measure_deltat = True
    #z = -40
    z = np.random.uniform(low=-43, high=-37, size=(n_agents,))


    
    len_exp = 100
    for nx in range(5 , 15):
        # airsim setup and takeoff
        # client.simPause(False)
        setup_drones(n_agents, names, client)
    
        # get drone locations and mean locations
        mean_x = 2.0
        mean_y = 0.0

        cost = 0
        ################################################################
        # option 1: two flocks colliding
        #x0, v0 = twoflocks(n_agents)

        #initial_v_dt = 8.0  # good for twoflocks()
        # initial_v_dt = 2.0 # better for the rest of the cases

        # option 2: two circles with inwards velocities
        # x0, v0 = circle(N)

        # option 3: tight grid of agents
        # x0 = grid(N)

        # option 4: random initialization as in training
        #states = benchmark[nx,:,:]
        states = problem.initialize()
        trace = {}
        trace['init_state'] = states
        trace['z'] = z

        x0 = states[:,0:3]
        v0 = states[:,2:4]
        initial_v_dt = 2.0 
        ######################################################################
    
        # scale positions and velocities
        x0 = x0 * init_scale
        if v0 is not None:
            v0 = v0 * init_scale

        display_msg(client, 'Moving to new positions...')

        send_loc_commands(client, n_agents, x0, home, mean_x, mean_y, z, names)
        if v0 is not None:
            send_velocity_commands(client, n_agents, v0, z, initial_v_dt, names)

        
        display_msg(client, 'Flocking...')
        history_imgs = np.zeros((n_agents, 3, 128 ))

        if isinstance(problem, FlockingProblem):
            x_agg3 = np.zeros((problem.n_nodes, problem.nx * problem.filter_len, problem.n_pools))
        else:
            x_agg3 = np.zeros((problem.n_agents * problem.nx, problem.episode_len))

        cost = 0 
        states = np.zeros((n_agents, 4, len_exp))
        costs = np.zeros((1, len_exp))
        for t in range(0, len_exp):
            xt1 = getStates(n_agents, names, client, home) / scale  # get drone locations and velocities
            ut1_gt = problem.controller(xt1)
            states[:,:, t] = xt1
            image = get_imgs(client, n_agents, names, model_yolo, device)
            if t == 0:
                history_imgs[:, 0, : ] = image[:, 0, : ]
                history_imgs[:, 1, : ] = image[:, 0, : ]
                history_imgs[:, 2, : ] = image[:, 0, : ]
            else:
                history_imgs = np.concatenate((image, history_imgs), axis=1)
                history_imgs = np.delete(history_imgs, [3], axis=1)    

            new_state = get_feats(client, n_agents, names, model_feat, history_imgs[:, :, :])
            new_state = np.clip(new_state, -30, 30)
            a_net = problem.get_connectivity(xt1)
            new_agg = problem.get_comms(problem.get_features(x_agg3[:, :, 0]), a_net)
            new_agg = problem.pooling[0](new_agg, axis=1)
            new_agg = new_agg.reshape((new_agg.shape[0], new_agg.shape[-1]))
            new_feat = np.concatenate((new_state, new_agg), axis=1)
            x_agg3[:, :, 0] = np.clip(new_feat, -100, 100)
            x_agg3_t = np.clip(x_agg3.reshape((problem.n_nodes, problem.filter_len * problem.nx * problem.n_pools)), -100, 100)

            record_images(client, n_agents, xt1, ut1_gt, names, nx, t, x_agg3_t, new_state)      

            #x_agg3_t = x_agg3.reshape((problem.n_nodes, problem.filter_len * problem.nx * problem.n_pools))

            x_agg3_t = torch.from_numpy(x_agg3_t).float().cuda()
            ut1 = model_agg(x_agg3_t).data.cpu().numpy().reshape(problem.n_nodes, problem.nu)


            if t == 0:
                init_cost = problem.instant_cost(xt1, ut1) 
            current_cost = problem.instant_cost(xt1, ut1) * problem.dt

            costs[:,t] = current_cost
            cost = cost + current_cost
              
            new_vel = (ut1 * true_dt + xt1[:, 2:4]) * scale
            
            # random pertubt z for visibility
            z = np.random.uniform(low=-43, high=-37, size=(n_agents,))


            send_velocity_commands(client, n_agents, new_vel, z, duration, names)
            print('current time step is ' + str(t) + ' , current cost is ' + str(current_cost))
            trace['states'] = states
            trace['costs'] = costs
            f = open('./trace/gt_' + str(nx) + '_' + str(t)+'.pkl',"wb")
            pickle.dump(trace,f)
            f.close()

       
        print('final_cost = ' + str(cost))    
        client.reset()
        trace['states'] = states
        trace['costs'] = costs
        f = open('./trace/gt' + str(nx) + '.pkl',"wb")
        pickle.dump(trace,f)
        f.close()
    





if __name__ == '__main__':
    main()
