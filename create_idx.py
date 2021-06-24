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

parser = argparse.ArgumentParser("createIndex")
parser.add_argument('--n-times', type=int, default=100, help='number of time steps for each exp')
parser.add_argument('--n-agents', type=int, default=50, help='number of agents steps for each exp')
parser.add_argument('--n-exp', type=int, default=15, help='number of total experiments')
parser.add_argument('--start-idx', type=int, default=0, help='start index of experiments')
parser.add_argument('--K', type=int, default=3, help='filter length')
parser.add_argument('--exp-name', type=str, default='/home/tkhu/Documents/AirSim/exp1104', help='root path to experiments')
args = parser.parse_args()


def main():

    dic = {}
    print('create index')
    n_time = args.n_times
    n_drone = args.n_agents
    counter = 0

    problem = pickle.load(open('problem.pkl', 'rb')) # store problem formulation
    problem.n_nodes = args.n_agents
    problem.filter_len = args.K
    #problem.comm_radius = args.comm_radius
    counter = 0
    filter_length = args.K
    max_accel = 10000
    # processing stable images
    root_data = os.path.join(args.exp_name, 'states')
    root_imgs = os.path.join(args.exp_name, 'imgs')
    for i in range(args.start_idx, args.start_idx + args.n_exp):
        a_nets_com15 = np.zeros((n_drone, n_drone, filter_length))
        a_nets_com25 = np.zeros((n_drone, n_drone, filter_length))
        a_nets_com35 = np.zeros((n_drone, n_drone, filter_length))

        for j in range(0, n_time):
            cur_dic = {}
            xt1 = np.zeros((n_drone, 4))
            for m in range(1, n_drone+1): 
                file_name = root_data + '/exp' + str(i) + '_time' + str(j) + '_Drone' + str(m) + '.txt'
                x_file = open(file_name, "r")
                xt = np.asarray((x_file.read().split()))
                xt1[m-1, :] = xt

        
            ut1 = problem.controller(xt1) # ground truth
            ut1 = np.clip(ut1, a_min=-max_accel, a_max=max_accel)
           

            problem.comm_radius = 1.5
            a_net_com15 = problem.get_connectivity(xt1) # connectivity
            problem.comm_radius = 2.5
            a_net_com25 = problem.get_connectivity(xt1) # connectivity
            problem.comm_radius = 3.5
            a_net_com35 = problem.get_connectivity(xt1) # connectivity


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
                    a_nets_com15[:, :, f] = a_net_com15
            else:
                a_nets_com15 = np.concatenate((a_nets_com15, np.expand_dims(a_net_com15, axis=2)), axis=2)
                a_nets_com15 = np.delete(a_nets_com15, [0], axis=2)    


            if j == 0:
                for f in range(filter_length):
                    a_nets_com35[:, :, f ] = a_net_com35
            else:
                a_nets_com35 = np.concatenate((a_nets_com35, np.expand_dims(a_net_com35, axis=2)), axis=2)
                a_nets_com35 = np.delete(a_nets_com35, [0], axis=2)    

            if j == 0:
                for f in range(filter_length):
                    a_nets_com25[:, :, f ] = a_net_com25
            else:
                a_nets_com25 = np.concatenate((a_nets_com25, np.expand_dims(a_net_com25, axis=2)), axis=2)
                a_nets_com25 = np.delete(a_nets_com25, [0], axis=2)    







            cur_dic['x_img_paths'] = img_path
            cur_dic['a_nets_com15'] =   a_nets_com15
            cur_dic['a_nets_com25'] =   a_nets_com25
            cur_dic['a_nets_com35'] =   a_nets_com35
            cur_dic['actions'] =  ut1
            dic[counter] = cur_dic
            print(counter)
            counter += 1


    f = open("./airsim_dataset_K{}.pkl".format(args.K),"wb")
    pickle.dump(dic,f)
    f.close()

if __name__ == "__main__":
    main()


