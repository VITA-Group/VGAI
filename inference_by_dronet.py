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
import joint_network as models
from scipy.spatial.distance import pdist, squareform

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser("test")
parser.add_argument('--n-times', type=int, default=100, help='number of time steps for each exp')
#parser.add_argument('--mode', type=str, default='optimal', choices=['optimal', 'vis_dagnn', 'vis_grnn'])
parser.add_argument('--model-path', type=str, default='checkpoint_all_vis_24_latest.tar', help='path to save the model')
parser.add_argument('--start-idx', type=int, default=0, help='start index of experiments')
parser.add_argument('--scale', type=float, default=6.0, help='scale factor for airsim environments')
parser.add_argument('--vinit', type=float, default=3.0, help='maximum intial velocity')
parser.add_argument('--radius', type=float, default=1.5, help='communication radius')
parser.add_argument('--K', type=int, default=3, help='number of filter length')
parser.add_argument('--F', type=int, default=24, help='number of feature dimension')
parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')
parser.add_argument('--comm-model', default='disk', choices=['disk', 'knn'], help='communication model')
parser.add_argument('--K-neighbor', type=int, default=10, help='number of KNN neighbors')
parser.add_argument('--noise', type=str, default=None, choices=['None', 'gaussian', 'blur'])

parser.add_argument('arch', default='vis_dagnn',
                        choices=model_names + ['optimal', 'local'],
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: vis_dagnn)')
args = parser.parse_args()
np.random.seed(args.seed)
def main():
    ## load model
    #problem.r_max = min(max(problem.r_max * (args.radius / 1.5), 5.0), 6.0)

    if args.arch not in ['optimal', 'local']:
        model = models.__dict__[args.arch](n_vis_out=args.F, K=args.K)
        params = torch.load(args.model_path)['state_dict']
        model.load_state_dict(params)
        model = torch.nn.DataParallel(model).cuda()
        model.eval()

    problem = pickle.load(open('problem.pkl', 'rb'))
    problem.comm_radius = args.radius


    problem.r_max = min(max(problem.r_max * (args.radius / 1.5), 5.0), 6.0)
    # parse settings file with drone names and home locations
    fname = '/home/tkhu/Documents/AirSim/settings.json'
    names, home = parse_settings(fname)
    n_agents = len(names)
    problem.n_nodes = n_agents
    problem.r_max = problem.r_max * (n_agents / 50.0)
    print('number of agents = {}'.format(n_agents))
    # use centralized controller
    problem.centralized_controller = True

    # rescale locations and velocities to avoid changing potential function
    scale = args.scale
    init_scale = args.scale

    # duration of velocity commands
    duration = 0.01  # duration of each control action used for training
    true_dt = 1.0 / 7.5  # average of actual measurements

    #duration = 2.0  # duration of each control action used for training
    #true_dt = 2.0  # average of actual measurements

    # make the problem easier because of slow control loop frequency
    v_max = args.vinit
    problem.v_max = v_max
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



    
    len_exp = args.n_times
    if args.comm_model == 'knn':
        trace_dir = '{}/vinit{}_scale{}_F{}_K{}_radius_{}_N{}_{}_neighbor{}_seed{}'.format(args.arch, int(v_max), scale, args.F, args.K, args.radius, n_agents, args.comm_model, args.K_neighbor, args.seed)
    else:
        trace_dir = '{}/vinit{}_scale{}_F{}_K{}_radius_{}_N{}_{}_seed{}'.format(args.arch, int(v_max), scale, args.F, args.K, args.radius, n_agents, args.comm_model, args.seed)

    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)
    # airsim setup and takeoff
    # client.simPause(False)
    setup_drones(n_agents, names, client)

    # get drone locations and mean locations
    mean_x = 2.0
    mean_y = 0.0
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
    states = problem.initialize(args.seed)
    while(True):
        z_loc = np.zeros((n_agents, 3))
        z = np.random.uniform(low=-38, high=-41, size=(n_agents,))
        z_loc[:, 0] = z
        z_loc[:, 1:] = states[:, 0:2]
        z_dist = squareform(pdist(z_loc.reshape((n_agents, 3)), 'euclidean'))
        z_dist = z_dist + 1000 * np.eye(n_agents)
        min_dist = np.min(np.min(z_dist))
        if min_dist > 0.4:
            break
    #print('z = {}'.format(z[0:5]))
    #print('states = {}'.format(states[0:5, :]))
    



    #problem.comm_radius = 2.5
    #states = initialize(n_agents, problem.v_bias , problem.v_max, 1.5, problem.r_max)
    trace = {}
    trace['init_state'] = states
    trace['z'] = z
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
        display_msg(client, 'Flocking...')
    if args.arch == 'loc_dagnn':
        x_imgs = np.zeros((n_agents, 6, args.K))
    else:
        x_imgs = np.zeros((n_agents, 3, 144, 1024, args.K))
    a_nets = np.zeros((n_agents, n_agents, args.K))
    cost = 0 
    states = np.zeros((n_agents, 4, len_exp + 1))
    costs = np.zeros((1, len_exp))
    if args.arch == 'vis_grnn':
        print('running vision-based GRNN')
        hidden_states = torch.from_numpy(np.zeros((n_agents, args.F, args.K))).float().cuda()



    for t in range(0, len_exp):
        xt1, _= getStates(n_agents, names, client, home)  # get drone locations and velocities
        xt1 = xt1 / scale
        states[:,:, t] = xt1
        trace['states'] = states
        trace['costs'] = costs
        f = open(os.path.join(trace_dir, 'timeSteps_' + str(t) + '.pkl'),"wb")
        pickle.dump(trace,f)
        f.close()
        if args.arch == 'vis_dagnn' or args.arch == 'vis_grnn':
            while(True):
                imgs = get_imgs(client, n_agents, names)
                if type(imgs) != type(None):
                    break
            if args.noise == 'gaussian':
                client.simPause(True)
                gaussian_imgs = np.zeros_like(imgs)
                for cur_idx in range(imgs.shape[0]):
                    gaussian_imgs[cur_idx, :, :, :] =  noisy(imgs[cur_idx, :, :, :])
                imgs = gaussian_imgs
                client.simPause(False)
            elif args.noise == 'blur':
                client.simPause(True)
                imgs = np.swapaxes(imgs, 1, 3)
                blur_imgs = np.zeros_like(imgs)
                for cur_idx in range(imgs.shape[0]):
                    tmp_img = imgs[cur_idx, :, :, :]
                    blur_imgs[cur_idx, :, :, :] =  cv2.blur(tmp_img, (5,5))
                imgs = blur_imgs
                imgs = np.swapaxes(imgs, 1, 3)
                client.simPause(False)

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
        if args.arch == 'loc_dagnn':
            x_features = problem.get_x_features(xt1)
            new_state = problem.get_comms(x_features, a_net)
            new_state = problem.pooling[0](new_state, axis=1)
            new_state = new_state.reshape((new_state.shape[0], new_state.shape[-1]))
            if t == 0:
                for k in range(args.K):
                    x_imgs[:, :, k ] = new_state
            else:
                x_imgs = np.concatenate((x_imgs, np.expand_dims(new_state, axis=2)), axis=2)
                x_imgs = np.delete(x_imgs, [0], axis=2)  

        elif args.arch not in ['optimal', 'local']:
            if t == 0:
                for k in range(args.K):
                    x_imgs[:, :, :, :, k ] = imgs
            else:
                x_imgs = np.concatenate((x_imgs, np.expand_dims(imgs, axis=4)), axis=4)
                x_imgs = np.delete(x_imgs, [0], axis=4) 
        x_imgs_t = torch.from_numpy(x_imgs).float().cuda()
        a_nets_t = torch.from_numpy(a_nets).float().cuda()

        if args.arch =='vis_grnn':
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

        elif args.arch == 'local':
            problem.centralized_controller = False
            ut1 = problem.controller(xt1)
            #print('local controller')
 
        else:
            problem.centralized_controller = True
            ut1 = problem.controller(xt1)
            #print('optimal controller')


        
        if t == 0:
            init_cost = problem.instant_cost(xt1, ut1) 
        current_cost = problem.instant_cost(xt1, ut1)
        costs[:,t] = current_cost
        cost = cost + current_cost
          
        new_vel = (ut1 * true_dt + xt1[:, 2:4]) * scale
        client.simPause(False)
        send_velocity_commands(client, n_agents, new_vel, z, duration, names)
        print('current time step is ' + str(t) + ' , current cost is ' + str(current_cost))

    print('final_cost = ' + str(cost))    
    client.reset()
    trace['states'] = states
    trace['costs'] = costs
    f = open(os.path.join(trace_dir, 'timeSteps_' + str(t) + '.pkl'),"wb")
    pickle.dump(trace,f)
    f.close()


def noisy(image):
      row,col,ch= image.shape
      mean = 0
      var = 100
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy


if __name__ == '__main__':
    main()
