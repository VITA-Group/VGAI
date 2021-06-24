from __future__ import print_function, division
import pickle
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import pickle
import cv2
import torchvision
import argparse
import joint_network as models
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser("createIndex")
parser.add_argument('--n-times', type=int, default=100, help='number of time steps for each exp')
parser.add_argument('--n-agents', type=int, default=50, help='number of agents steps for each exp')
parser.add_argument('--n-exp', type=str, default='15', help='number of total experiments')
parser.add_argument('--start-idx', type=str, default=0, help='start index of experiments')
parser.add_argument('--K', type=int, default=3, help='filter length')
parser.add_argument('--exp-name', type=str, default='/home/tkhu/Documents/AirSim/exp1104', help='root path to experiments')

parser.add_argument('--vinit', type=float, default=3.0, help='maximum intial velocity')
parser.add_argument('--radius', type=float, default=1.5, help='communication radius')
parser.add_argument('--F', type=int, default=24, help='number of feature dimension')
parser.add_argument('--comm-model', default='disk', choices=['disk', 'knn'], help='communication model')
parser.add_argument('--K-neighbor', type=int, default=10, help='number of KNN neighbors')

parser.add_argument('--mode', type=str, default='optimal', choices=['optimal', 'local', 'loc_dagnn', 'vis_dagnn', 'vis_grnn', 'loc_grnn'])

parser.add_argument('--arch', default='vis_dagnn',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: vis_dagnn)')

args = parser.parse_args()


def main():

    exp_name_list = list(map(str, args.exp_name.split(',')))
    start_idx_list = list(map(str, args.start_idx.split(',')))
    n_exp_list = list(map(str, args.n_exp.split(',')))
    assert len(exp_name_list) == len(start_idx_list) == len(n_exp_list), "length of experiments should be the same"
    n_folder = len(n_exp_list)
    dic = {}
    counter = 0
    problem = pickle.load(open('problem.pkl', 'rb')) # store problem formulation
    problem.n_nodes = args.n_agents
    problem.filter_len = args.K
    filter_length = args.K
    problem.comm_radius = args.radius
    n_time = args.n_times
    n_drone = args.n_agents
    max_accel = 10
    print(exp_name_list)
    print(start_idx_list)
    print(n_exp_list)
    for n_exp, exp_name, start_idx in zip(n_exp_list, exp_name_list, start_idx_list):
        print('n_exp = {}, exp_name = {}, start_idx = {}'.format(n_exp, exp_name, start_idx)) 
        # processing stable images
        start_idx = int(start_idx)
        n_exp = int(n_exp)
        root_data = os.path.join(exp_name, 'states')
        root_imgs = os.path.join(exp_name, 'imgs')
        for i in range(start_idx, start_idx + n_exp):
            a_nets = np.zeros((n_drone, n_drone, filter_length))
            x_locs = np.zeros((n_drone, 4, filter_length))
            x_aggs = np.zeros((n_drone, 6, filter_length))
            for j in range(0, n_time):
                cur_dic = {}
                xt1 = np.zeros((n_drone, 4))
                for m in range(1, n_drone+1): 
                    file_name = root_data + '/exp' + str(i) + '_time' + str(j) + '_Drone' + str(m) + '.txt'
                    x_file = open(file_name, "r")
                    xt = np.asarray((x_file.read().split()))
                    xt1[m-1, :] = xt

                #print('x featues shape = {}'.format(x_features.shape))  
                   
    
                ut1 = problem.controller(xt1) # ground truth
                ut1 = np.clip(ut1, a_min=-max_accel, a_max=max_accel)
                if args.comm_model == 'disk':
                    a_net = problem.get_connectivity(xt1) 
                elif args.comm_model == 'knn':
                    a_net = problem.get_knn_connectivity(xt1, args.K_neighbor) 
                #print(np.sum(a_net))


                x_features = problem.get_x_features(xt1)
                new_state = problem.get_comms(x_features, a_net)
                new_state = problem.pooling[0](new_state, axis=1)
                new_state = new_state.reshape((new_state.shape[0], new_state.shape[-1]))


                #print('new state shape = {}'.format(new_state.shape))  


                img_path = {}
                for n in range(filter_length):
                    for m in range(1, n_drone+1): 
                        if j - n >= 0:
                            f_img_file = root_imgs + '/exp' + str(i) + '_time' + str(j-n) + '_Drone' + str(m) + '_0.png'
                            l_img_file = root_imgs + '/exp' + str(i) + '_time' + str(j-n) + '_Drone' + str(m) + '_1.png'
                            r_img_file = root_imgs + '/exp' + str(i) + '_time' + str(j-n) + '_Drone' + str(m) + '_2.png'
                            b_img_file = root_imgs + '/exp' + str(i) + '_time' + str(j-n) + '_Drone' + str(m) + '_3.png'
                            img_path['time-{}_drone_{}'.format(n, m)] = [f_img_file, l_img_file, r_img_file, b_img_file]
                        else:
                            print('time_idx = {}'.format(j))
                            f_img_file = root_imgs + '/exp' + str(i) + '_time' + str(j) + '_Drone' + str(m) + '_0.png'
                            l_img_file = root_imgs + '/exp' + str(i) + '_time' + str(j) + '_Drone' + str(m) + '_1.png'
                            r_img_file = root_imgs + '/exp' + str(i) + '_time' + str(j) + '_Drone' + str(m) + '_2.png'
                            b_img_file = root_imgs + '/exp' + str(i) + '_time' + str(j) + '_Drone' + str(m) + '_3.png'
                            img_path['time-{}_drone_{}'.format(n, m)] = [f_img_file, l_img_file, r_img_file, b_img_file]
                           
                            if not os.path.isfile(f_img_file):
                                print('{} not exit'.format(f_img_file))
                            if not os.path.isfile(l_img_file):
                                print('{} not exit'.format(l_img_file))
                            if not os.path.isfile(r_img_file):
                                print('{} not exit'.format(r_img_file))
                            if not os.path.isfile(b_img_file):
                                print('{} not exit'.format(b_img_file))
    
                if j == 0:
                    for f in range(filter_length):
                        a_nets[:, :, f] = a_net

                    for f in range(filter_length):
                        x_locs[:, :, f] = xt1

                    for f in range(filter_length):
                        x_aggs[:, :, f] = new_state

                else:
                    a_nets = np.concatenate((a_nets, np.expand_dims(a_net, axis=2)), axis=2)
                    a_nets = np.delete(a_nets, [0], axis=2)    


                    x_locs = np.concatenate((x_locs, np.expand_dims(xt1, axis=2)), axis=2)
                    x_locs = np.delete(x_locs, [0], axis=2)    

                    x_aggs = np.concatenate((x_aggs, np.expand_dims(new_state, axis=2)), axis=2)
                    x_aggs = np.delete(x_aggs, [0], axis=2)    

    
                cur_dic['x_img_paths'] = img_path
                cur_dic['a_nets'] =   a_nets
                cur_dic['actions'] =  ut1
                cur_dic['x_locs'] = x_locs
                cur_dic['x_aggs'] = x_aggs
                dic[counter] = cur_dic
                print(counter)
                counter += 1
    
    if args.comm_model == 'disk':
        file_name = '{}_K_{}_n_vis_{}_R_{}_vinit_{}_comm_model_{}.pkl'.format(args.mode, args.K, args.F, args.radius, args.vinit, args.comm_model)
    else:
        file_name = '{}_K_{}_n_vis_{}_vinit_{}_comm_model_{}_K_neighbor_{}.pkl'.format(args.mode, args.K, args.F, args.vinit, args.comm_model, args.K_neighbor)
    f = open(file_name,"wb")
    pickle.dump(dic,f)
    f.close()

if __name__ == "__main__":
    main()


