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
from scipy.spatial.distance import pdist, squareform
import  joint_network as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))
print(model_names)
parser = argparse.ArgumentParser("collectData")
parser.add_argument('--n-times', type=int, default=100, help='number of time steps for each exp')
parser.add_argument('--mode', type=str, default='optimal', choices=['optimal', 'local', 'loc_dagnn', 'vis_dagnn', 'vis_grnn', 'vis_yolo_dagnn', 'loc_grnn'])
parser.add_argument('--model-path', type=str, default='None', help='path to save the model')
parser.add_argument('--exp-name', type=str, default='/home/tkhu/Documents/AirSim/exp1104', help='root path to experiments')
parser.add_argument('--n-exp', type=int, default=1, help='number of experiments')
parser.add_argument('--start-idx', type=int, default=0, help='start index of experiments')
parser.add_argument('--scale', type=float, default=6.0, help='scale factor for airsim environments')
parser.add_argument('--vinit', type=float, default=3.0, help='maximum intial velocity')
parser.add_argument('--radius', type=float, default=1.5, help='communication radius')
parser.add_argument('--K', type=int, default=3, help='number of filter length')
parser.add_argument('--F', type=int, default=24, help='number of feature dimension')
parser.add_argument('--arch', metavar='ARCH', default='vis_dagnn',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: vis_dagnn)')

parser.add_argument('--comm-model', default='disk', choices=['disk', 'knn'], help='communication model')

parser.add_argument('--K-neighbor', type=int, default=10, help='number of KNN neighbors')

parser.add_argument('--beta', type=float, default=0.33, help='betta for dagger algorithm')


args = parser.parse_args()


def main():
    problem = pickle.load(open('problem.pkl', 'rb'))
    problem.comm_radius = args.radius
    fname = '/home/tkhu/Documents/AirSim/settings.json'
    names, home = parse_settings(fname)
    n_agents = len(names)
    problem.n_nodes = n_agents
    problem.filter_len = args.K
    print('mode = {}'.format(args.mode))
    problem.r_max = min(max(problem.r_max * (args.radius / 1.5), 5.0), 6.0)
    isDagger = True

    if args.mode != 'optimal':
        if args.arch != 'loc_dagnn':
            model = models.__dict__[args.arch](n_vis_out=args.F, K=args.K)
            params = torch.load(args.model_path)['state_dict']
            model.load_state_dict(params)
            model = torch.nn.DataParallel(model).cuda()
            model.eval()
            print('load model @ {}'.format(args.model_path))
        else:
            model = models.__dict__[args.arch](K=args.K)
            model = model.cuda()
        model = model.eval()
    #####################
    # use centralized controller
    problem.centralized_controller = True

    # rescale locations and velocities to avoid changing potential function
    scale = args.scale
    init_scale = args.scale

    # duration of velocity commands
    duration = 0.01  # duration of each control action used for training
    true_dt = 1.0 / 7.5  # average of actual measurements

    # make the problem easier because of slow control loop frequency
    problem.v_max = args.vinit
    problem.v_max = problem.v_max * 0.075
    problem.v_bias = problem.v_bias * 0.3



    


    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    display_msg(client, 'Initializing...')

    measure_deltat = True
    #z = -40
    #while(True):
    #    z_loc = np.zeros((n_agents, 2))
    #    z = np.random.uniform(low=-37, high=-43, size=(n_agents,))
    #    z_loc[:, 0] = z
    #    z_dist = squareform(pdist(z_loc.reshape((n_agents, 2)), 'euclidean'))
    #    z_dist = z_dist + 1000 * np.eye(n_agents)
    #    min_dist = np.min(np.min(z_dist))
    #    if min_dist > 0.005:
    #        break
        #else:
        #   print('min_dist ={}'.format(min_dist))

 


    for nx in range(args.start_idx ,args.start_idx + args.n_exp):
        trace = {}
        print('exp ' + str(nx))

        # airsim setup and takeoff
        # client.simPause(False)
        setup_drones(n_agents, names, client)
    
        # get drone locations and mean locations
        xt1, yaws = getStates(n_agents, names, client, home)
        mean_x = np.mean(xt1[:, 0])
        mean_y = np.mean(xt1[:, 1])

        #cost = 0
    
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
        states = problem.initialize()
        while(True):
            z_loc = np.zeros((n_agents, 3))
            z = np.random.uniform(low=-38, high=-41, size=(n_agents,))
            z_loc[:, 0] = z
            z_loc[:, 1:] = states[:, 0:2]
            z_dist = squareform(pdist(z_loc.reshape((n_agents, 3)), 'euclidean'))
            z_dist = z_dist + 1000 * np.eye(n_agents)
            min_dist = np.min(np.min(z_dist))
            #print('min dist = {}'.format(min_dist))
            if min_dist > 0.4:
                break
    

        x0 = states[:,0:2]
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

        
        if args.mode != 'optimal':
            if args.mode == 'loc_dagnn' or args.mode == 'loc_grnn':
                x_imgs = np.zeros((n_agents, 6, args.K))
                a_nets = np.zeros((n_agents, n_agents, args.K))
                if args.arch == 'loc_grnn': 
                    print('running location-based GRNN')
                    hidden_states = torch.from_numpy(np.zeros((n_agents, 6, args.K))).float().cuda()
            else:
                x_imgs = np.zeros((n_agents, 3, 144, 1024, args.K))
                a_nets = np.zeros((n_agents, n_agents, args.K))

                if args.arch == 'vis_grnn': 
                    print('running vision-based GRNN')
                    hidden_states = torch.from_numpy(np.zeros((n_agents, args.F, args.K))).float().cuda()

        display_msg(client, 'Flocking...')
        len_exp = args.n_times
        



        if isinstance(problem, FlockingProblem):
            x_agg3 = np.zeros((problem.n_nodes, problem.nx * problem.filter_len, problem.n_pools))
        else:
            x_agg3 = np.zeros((problem.n_agents * problem.nx, problem.episode_len))


        cost = 0
        states = np.zeros((n_agents, 4, len_exp))
        costs = np.zeros((1, len_exp))

        for t in range(0, len_exp):

            xt1, yaws = getStates(n_agents, names, client, home)  # get drone locations and velocities

            xt1 = xt1 / scale
            states[:,:, t] = xt1
            

            # use designed controller
            ut1_optimal = problem.controller(xt1)
            x_features = problem.get_x_features(xt1)
            a_net = problem.get_connectivity(xt1)
            new_state = problem.get_comms(x_features, a_net)
            new_state = problem.pooling[0](new_state, axis=1)
            new_state = new_state.reshape((new_state.shape[0], new_state.shape[-1]))
            new_agg = problem.get_comms(problem.get_features(x_agg3[:, :, 0]), a_net)
            new_agg = problem.pooling[0](new_agg, axis=1)
            new_agg = new_agg.reshape((new_agg.shape[0], new_agg.shape[-1]))
            new_feat = np.concatenate((new_state, new_agg), axis=1)
            x_agg3[:, :, 0] = new_feat
            x_agg3_t = x_agg3.reshape((problem.n_nodes, problem.filter_len * problem.nx * problem.n_pools))


            if args.mode != 'optimal':
                if args.arch != 'loc_dagnn' or args.arch != 'loc_grnn':
                    while(True):
                        imgs = get_imgs(client, n_agents, names)
                        if type(imgs) != type(None):
                            break

                client.simPause(True)
                if args.comm_model == 'knn':
                    a_net = problem.get_knn_connectivity(xt1, args.K_neighbor)
                else:
                    a_net = problem.get_connectivity(xt1)
                if t == 0:
                    for k in range(args.K):
                        a_nets[:, :, k ] = a_net
                else:
                    a_nets = np.concatenate((a_nets, np.expand_dims(a_net, axis=2)), axis=2)
                    a_nets = np.delete(a_nets, [0], axis=2) # 50 x 50 x 3
                a_nets[a_nets != a_nets] = 0
        
                if args.arch == 'loc_dagnn' or args.arch == 'loc_grnn':
                    #new_state = np.clip(new_state, -30, 30)
                    if t == 0:
                        for k in range(args.K):
                            x_imgs[:, :, k ] = new_state
                    else:
                        x_imgs = np.concatenate((x_imgs, np.expand_dims(new_state, axis=2)), axis=2)
                        x_imgs = np.delete(x_imgs, [0], axis=2) 
                else:
                    if t == 0:
                        for k in range(args.K):
                            x_imgs[:, :, :, :, k ] = imgs
                    else:
                        x_imgs = np.concatenate((x_imgs, np.expand_dims(imgs, axis=4)), axis=4)
                        x_imgs = np.delete(x_imgs, [0], axis=4) 
                x_imgs_t = torch.from_numpy(x_imgs).float().cuda()
                a_nets_t = torch.from_numpy(a_nets).float().cuda()
        
                if args.arch =='vis_grnn' or args.arch == 'loc_grnn':
                    current_hidden_states = hidden_states[:, :, 0]
                    current_hidden_states = current_hidden_states.data
                    hidden_states, ut1 = model(x_imgs_t, a_nets_t, current_hidden_states)
                    ut1 = ut1.data.cpu().numpy().reshape(problem.n_nodes, problem.nu)
                elif args.arch =='vis_dagnn':
                    _, ut1 = model(x_imgs_t, a_nets_t)
                    ut1 = ut1.data.cpu().numpy().reshape(problem.n_nodes, problem.nu)
                elif args.arch =='loc_dagnn':
                    ut1 = model(x_imgs_t, a_nets_t)
                    ut1 = ut1.data.cpu().numpy().reshape(problem.n_nodes, problem.nu)
                    ut1 = np.clip(ut1, -1, 1)


                else:
                    ut1 = problem.controller(xt1)
                    isDagger = False
    
                client.simPause(False)
    

            if args.mode == 'optimal':
                ut1 = ut1_optimal
               
            if t == 0:
                init_cost = problem.instant_cost(xt1, ut1) 
            current_cost = problem.instant_cost(xt1, ut1)
            while(True):
                isrecord = record_images(client, n_agents, xt1, ut1_optimal, names, nx, t, x_agg3_t, new_state, args.exp_name)      
                print('isRecord = {}'.format(isrecord))
                if isrecord == True:
                    break
            if isDagger:
                new_ut1 = args.beta * ut1_optimal + (1 - args.beta) * ut1
                new_vel = (new_ut1 * true_dt + xt1[:, 2:4]) * scale
                send_velocity_commands(client, n_agents, new_vel, z, duration, names)

            else:
                new_vel = (ut1 * true_dt + xt1[:, 2:4]) * scale
                send_velocity_commands(client, n_agents, new_vel, z, duration, names)
            print('current time step is ' + str(t) + ' , current cost is ' + str(current_cost))
            
        client.reset()
        trace['states'] = states
        trace['costs'] = costs
        if not os.path.exists('./trace'):
            os.makedirs('./trace')
        if args.mode == 'optimal':
            f = open('./trace/optimal_controller_' + str(nx) + '.pkl',"wb")
        else:
            trace_dir = 'trace/{}_vinit{}_scale{}_F{}_K{}_radius_{}_N{}_exp{}.pkl'.format(args.arch, \
                                                           int(args.vinit), args.scale, args.F, args.K, args.radius, n_agents, str(nx))
            f = open(trace_dir,"wb")

        pickle.dump(trace,f)
        f.close()
if __name__ == '__main__':
    main()
