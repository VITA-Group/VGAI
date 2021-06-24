from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
import cv2
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import random

from scipy.spatial.distance import pdist, squareform

class OneHopDataset(Dataset):
    def __init__(self, f_name=None, K=3):
        self.root_dir = './data' 
        if f_name:
            print('f name  = {}'.format(f_name))
            self.data = pickle.load(open(f_name, "rb" ))
        else:
            self.data = pickle.load(open('./airsim_dataset_F3.pkl', "rb" ))
        self.K = K

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_img_paths = self.data[index]['x_img_paths']
        fovs = []
        x_img = []
        for j in range(self.K):
            fovs = []
            for k in range(1, 51):
                img_files = x_img_paths['time-{}_drone_{}'.format(j, k)]
                f_img_file = img_files[0]
                l_img_file = img_files[1]
                r_img_file = img_files[2]
                b_img_file = img_files[3]
                f_image = cv2.imread(f_img_file)
                l_image = cv2.imread(l_img_file)
                r_image = cv2.imread(r_img_file)
                b_image = cv2.imread(b_img_file)
                if type(f_image) == type(None):
                    if j == 0:
                        img_files = x_img_paths['time-{}_drone_{}'.format(j+1, k)]
                        f_img_file = img_files[0]
                        l_img_file = img_files[1]
                        r_img_file = img_files[2]
                        b_img_file = img_files[3]
                        f_image = cv2.imread(f_img_file)
                        l_image = cv2.imread(l_img_file)
                        r_image = cv2.imread(r_img_file)
                        b_image = cv2.imread(b_img_file)
                    else:
                        img_files = x_img_paths['time-{}_drone_{}'.format(j-1, k)]
                        f_img_file = img_files[0]
                        l_img_file = img_files[1]
                        r_img_file = img_files[2]
                        b_img_file = img_files[3]
                        f_image = cv2.imread(f_img_file)
                        l_image = cv2.imread(l_img_file)
                        r_image = cv2.imread(r_img_file)
                        b_image = cv2.imread(b_img_file)

                fov = np.concatenate([f_image, l_image, r_image, b_image], axis=1)
                
                fov = fov.swapaxes(0, 2).swapaxes(1, 2)
                fovs.append(np.expand_dims(fov, axis=0))
            fovs = np.expand_dims(np.concatenate(fovs, axis=0), axis=4)
            x_img.append(fovs)
        x_img = x_img[::-1]
        x_img = np.concatenate(x_img, axis=4)
        #x_agg = np.clip(self.data[index]['x_agg'], -30, 30)
        #a_nets = self.data[index]['a_nets']
        #mylist = ['0', '1', '2']
        #choice = random.choice(mylist)
        #if choice == '0':
        #    a_nets = self.data[index]['a_nets_com15']
        #    #print('choose com 1.5')
        #elif choice == '1':
        #    a_nets = self.data[index]['a_nets_com25']
        #    #print('choose com 2.5')
        #else:
        #    a_nets = self.data[index]['a_nets_com35']
        x_locs = self.data[index]['x_locs'] # 50 x 4 x K
        n_drone = x_locs.shape[0]
        a_nets = np.zeros((n_drone, n_drone, self.K))
        for j in range(self.K):
            R = np.random.uniform(low=1.0, high=3.5, size=None)
            a_net = get_connectivity(x_locs[:, :, j], R)
            a_nets[:, :, j] = a_net

        a_nets[a_nets != a_nets] = 0
        actions = self.data[index]['actions']
        aggs = np.clip(self.data[index]['x_aggs'], -30, 30)
       

        sample = {'anets': a_nets, 'actions': actions,  'x_img': x_img, 'x_agg': aggs}

        return sample

def imread(path, device):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.transpose(2,0,1)
    img = torch.from_numpy(img).float().to(device)
    img, _ = pad_to_square(img, 0)
    img = resize(img, 416).unsqueeze(0)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    return Variable(img.type(Tensor))

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def get_connectivity(x, comm_radius):
    """
    Get the adjacency matrix of the network based on agent locations by computing pairwise distances using pdist
    Args:
        x (): current states of all agents

    Returns: adjacency matrix of network

    """
    n_nodes = x.shape[0]
    x_t_loc = x[:, 0:2]  # x,y location determines connectivity
    a_net = squareform(pdist(x_t_loc.reshape((n_nodes, 2)), 'euclidean'))
    a_net = (a_net < comm_radius).astype(float)
    np.fill_diagonal(a_net, 0)
    return a_net


def main():
    print('test dataloader')
    drone_dataset =  OneHopDataset(f_name='./optimal_K_3_n_vis_24_R_1.5_vinit_3.0_comm_model_disk.pkl')
    print(len(drone_dataset))
   
    droneTrainLoader = torch.utils.data.DataLoader(drone_dataset,batch_size=1,shuffle=True, num_workers=4)
    max_value = -float("Inf")
    for i_batch,sample_batched in enumerate(droneTrainLoader,0):
        #print('sample index = {}, agg shape = {}'.format(i_batch,sample_batched['x_agg'].shape))
        print('sample index = {}, action shape = {}'.format(i_batch,sample_batched['actions'].shape))
        print('sample index = {}, anet shape = {}'.format(i_batch,sample_batched['anets'].shape))
        print('sample index = {}, img shape = {}'.format(i_batch,sample_batched['x_img'].shape))

        print('sample index = {}, action = {}'.format(i_batch,sample_batched['actions'].shape))
        print('sample index = {}, x aggs = {}'.format(i_batch,sample_batched['x_agg'].shape))
      


if __name__ == "__main__":
    main()
