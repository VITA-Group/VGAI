#!/bin/bash
n_times=100
n_agents=50
n_exp=15
start_idx=0
filterLength=3
n_vis=24
radius=1.5
vinit=1.0
root_path='/ssd1/tkhu/AirSim/optimal_vinit_'$vinit'_R_'$radius
mode='optimal'
arch='vis_grnn'
comm_model='knn'
K_neighbor=10
#python grad_train_data_optimal_location.py --n-times $n_times --n-exp $n_exp --start-idx $start_idx --K $filterLength --exp-name $root_path --mode $mode --radius $radius --K-neighbor $K_neighbor
python prepare_training.py --n-times $n_times --n-agents $n_agents --n-exp $n_exp --start-idx $start_idx --K $filterLength --exp-name $root_path --K $filterLength --radius $radius --vinit $vinit --arch $arch --F $n_vis  --comm-model $comm_model --mode $mode --K-neighbor $K_neighbor
CUDA_VISIBLE_DEVICES=0 python train.py $arch  --K $filterLength --radius $radius --vinit $vinit  --F $n_vis --comm-model $comm_model --mode $mode --K-neighbor $K_neighbor




if [[ $comm_model == "disk" ]]
then
   model_path='checkpoint_all_'$mode'_'$arch'_K_'$filterLength'_n_vis_'$n_vis'_R_'$radius'_vinit_'$vinit'_comm_model_'$comm_model'.tar'
   root_path_aug='/ssd1/tkhu/AirSim/'$arch'_K_'$filterLength'_n_vis_'$n_vis'_R_'$radius'_vinit_'$vinit'_comm_model_'$comm_model
else
   model_path='checkpoint_all_'$mode'_'$arch'_K_'$filterLength'_n_vis_'$n_vis'_vinit_'$vinit'_comm_model_'$comm_model'_K_neighbor_'$K_neighbor'.tar'
   root_path_aug='/ssd1/tkhu/AirSim/'$arch'_K_'$filterLength'_n_vis_'$n_vis'_vinit_'$vinit'_comm_model_'$comm_model'_K_neighbor_'$K_neighbor
fi
 
mode='vis_grnn'
n_exp_aug=5
start_idx_aug=0
echo $root_path
echo $model_path
CUDA_VISIBLE_DEVICES=0 python grad_train_data_optimal_location.py --n-times $n_times --n-exp $n_exp_aug --start-idx $start_idx_aug --K $filterLength --radius $radius --vinit $vinit --arch $arch --F $n_vis --exp-name $root_path_aug --mode  $mode --model-path  $model_path --comm-model $comm_model --K-neighbor $K_neighbor

joint_exp_name=$root_path','$root_path_aug
joint_start_idx=$start_idx','$start_idx_aug
joint_n_exp=$n_exp','$n_exp_aug
python prepare_training.py --n-times $n_times --n-agents $n_agents --n-exp $joint_n_exp --start-idx $joint_start_idx --K $filterLength --exp-name $joint_exp_name --K $filterLength --radius $radius --vinit $vinit --arch $arch --F $n_vis  --comm-model $comm_model --mode $mode  --K-neighbor $K_neighbor
CUDA_VISIBLE_DEVICES=0 python train.py $arch  --K $filterLength --radius $radius --vinit $vinit  --F $n_vis --comm-model $comm_model --mode $mode --K-neighbor $K_neighbor



