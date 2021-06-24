import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

import joint_network as models
from dataset_loc import OneHopDataset
from torch.utils.data import Dataset, DataLoader
import shutil

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser("training")
parser.add_argument('arch', metavar='ARCH', default='vis_dagnn',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: vis_dagnn)')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=5, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--decreasing_lr', default='5,10,15', help='decreasing strategy')
parser.add_argument('--K', type=int, default=3, help='filter length')
parser.add_argument('--vinit', type=float, default=3.0, help='maximum intial velocity')
parser.add_argument('--radius', type=float, default=1.5, help='communication radius')
parser.add_argument('--F', type=int, default=24, help='number of feature dimension')
parser.add_argument('--comm-model', default='disk', choices=['disk', 'knn'], help='communication model')
parser.add_argument('--K-neighbor', type=int, default=10, help='number of KNN neighbors')
parser.add_argument('--mode', type=str, default='optimal', choices=['optimal', 'local', 'loc_dagnn', 'vis_dagnn', 'vis_grnn', 'loc_grnn'])



args = parser.parse_args()


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  model = models.__dict__[args.arch](n_vis_out=args.F, K=args.K)
  model = model.cuda()


  train_criterion = torch.nn.SmoothL1Loss()
  criterion = torch.nn.SmoothL1Loss() #torch.nn.MSELoss(reduction='mean')
  train_criterion = train_criterion.cuda()
  criterion = criterion.cuda()


  train_criterion = train_criterion.cuda()
  criterion = criterion.cuda()
  decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

  
  
  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=args.learning_rate,
      weight_decay=args.weight_decay
      )
  
  if args.comm_model == 'disk':
      f_name = '{}_K_{}_n_vis_{}_R_{}_vinit_{}_comm_model_{}.pkl'.format(args.mode, args.K, args.F, args.radius, args.vinit, args.comm_model)
  else:
      f_name = '{}_K_{}_n_vis_{}_vinit_{}_comm_model_{}_K_neighbor_{}.pkl'.format(args.mode, args.K, args.F, args.vinit, args.comm_model, args.K_neighbor)


  drone_dataset = OneHopDataset(f_name=f_name, K=args.K, R=args.radius)
  num_train = len(drone_dataset)
  indices = list(range(num_train))
  split = int(np.floor(0.9 * num_train))

  train_queue = torch.utils.data.DataLoader(drone_dataset,batch_size=args.batch_size, num_workers=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), pin_memory=True)
  
  valid_queue = torch.utils.data.DataLoader(drone_dataset,batch_size=1, num_workers=2, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]), pin_memory=True)
 
  print('Training Joint Network')
  #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
  for epoch in range(args.epochs):
    #scheduler.step()
    #logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    train_loss = train_joint(train_queue, model, train_criterion, optimizer)
    valid_loss = infer_joint(valid_queue, model, criterion)
    #checkpoint_path  =  'checkpoint_all_{}_{}_K_{}_n_vis_{}_R_{}_vinit_{}_comm_model_{}.tar'.format(args.mode, args.arch, args.K, args.F, args.radius, args.vinit, args.comm_model)
    #checkpoint_path = 'checkpoint_all_vis_{}_{}_latest.tar'.format(args.F, args.arch)
    if args.comm_model == 'disk':
        checkpoint_path  =  'checkpoint_all_{}_{}_K_{}_n_vis_{}_R_{}_vinit_{}_comm_model_{}.tar'.format(args.mode, args.arch, args.K, args.F, args.radius, args.vinit, args.comm_model)
    else:
        checkpoint_path  =  'checkpoint_all_{}_{}_K_{}_n_vis_{}_vinit_{}_comm_model_{}_K_neighbor_{}.tar'.format(args.mode, args.arch, args.K, args.F, args.vinit, args.comm_model, args.K_neighbor)

    if True: 
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss': valid_loss,
        }, filename=checkpoint_path)
        best_valid_loss = valid_loss
    print('epoch ' + str(epoch) + ' train loss ' + str(train_loss) + ' valid loss ' + str(valid_loss))

def train_joint(train_queue, model, criterion, optimizer):

  objs = AvgrageMeter()
  model.train()
  print('len of train queue = {}'.format(len(train_queue)))
  total_loss = 0
  for step, sample_batched in enumerate(train_queue, 0):
    #x_img = sample_batched['x_img'].float().cuda().squeeze(0)
    x_agg = sample_batched['x_agg'].float().cuda().squeeze(0)
    a_nets = sample_batched['anets'].float().cuda().squeeze(0)
    actions = sample_batched['actions'].float().cuda().squeeze(0)
    if args.arch == 'vis_grnn':
        input_state = torch.from_numpy(np.zeros((x_img.shape[0], args.F))).float().cuda()
        pred_agg, pred = model(x_img, a_nets, input_state)
    elif args.arch == 'loc_dagnn':
        pred = model(x_agg, a_nets)
    elif args.arch == 'loc_grnn':
        input_state = torch.from_numpy(np.zeros((x_agg.shape[0], 6))).float().cuda()
        pred_agg, pred = model(x_agg, a_nets, input_state)

    else:
        pred_agg, pred = model(x_img, a_nets)
    loss = criterion(pred, actions)
    #print('loss = {}'.format(loss))
    total_loss += loss
    if step % 1 == 0:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        total_loss = 0


    n = pred.size(0)   
    objs.update(loss.item(), n)
    if step % args.report_freq == 0:
        print('-----')
        print('train step ' + str(step) + ' loss ' + str(objs.avg))
        #print('pred = {}'.format(pred))
        #print('actions = {}'.format(actions))
        print('-----')
  
  return objs.avg



def infer_joint(valid_queue, model, criterion):
  objs = AvgrageMeter()
  model.eval()

  for step, sample_batched in enumerate(valid_queue, 0):
    #x_img = sample_batched['x_img'].float().cuda().squeeze(0)
    x_agg = sample_batched['x_agg'].float().cuda().squeeze(0)
    a_nets = sample_batched['anets'].float().cuda().squeeze(0)
    actions = sample_batched['actions'].float().cuda().squeeze(0)
    if args.arch == 'vis_grnn':
        input_state = torch.from_numpy(np.zeros((x_img.shape[0], args.F))).float().cuda()
        _, pred = model(x_img, a_nets, input_state)
    elif args.arch == 'loc_dagnn':
        pred = model(x_agg, a_nets)
    elif args.arch == 'loc_grnn':
        input_state = torch.from_numpy(np.zeros((x_agg.shape[0], 6))).float().cuda()
        pred_agg, pred = model(x_agg, a_nets, input_state)
    else:
        _, pred = model(x_img, a_nets)
    loss = criterion(pred, actions)


    
    n = pred.size(0)   
    objs.update(loss.item(), n)
    if step % args.report_freq == 0:
        print('-----')
        print('valid step ' + str(step) + ' loss ' + str(objs.avg))
        print('-----')

  return objs.avg



class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
  main() 

