#!/bin/bash
n_times=100
#n_agents=50
#n_exp=15
#start_idx=0
filterLength=3
#root_path='/home/tkhu/Documents/AirSim/exp1104'
n_vis=24
vint=3.0
radius=1.5
model_path='checkpoint_all_optimal_loc_dagnn_K_3_n_vis_24_R_1.5_vinit_1.0_comm_model_disk.tar'
comm_model='disk'
K_neighbor='25'
arch='optimal'
#f_name='./airsim_dataset_K'${filterLength}'.pkl'
#python grad_train_data_optimal_location.py --n-times $n_times --n-exp $n_exp --start-idx $start_idx --K $filterLength --exp-name $root_path 
#python create_idx.py --n-times $n_times --n-agents $n_agents --n-exp $n_exp --start-idx $start_idx --K $filterLength --exp-name $root_path 
seed=2
for i in {0..1}
do
CUDA_VISIBLE_DEVICES=0 python inference_by_dronet.py $arch --vinit $vint  --radius $radius --K $filterLength --F $n_vis --seed $seed --model-path $model_path --comm-model $comm_model --K-neighbor $K_neighbor
done

seed=3
for i in {0..1}
do
CUDA_VISIBLE_DEVICES=0 python inference_by_dronet.py $arch --vinit $vint  --radius $radius --K $filterLength --F $n_vis --seed $seed --model-path $model_path --comm-model $comm_model --K-neighbor $K_neighbor
done

#CUDA_VISIBLE_DEVICES=0 python inference_by_dronet.py $arch --vinit $vint  --radius $radius --K $filterLength --F $n_vis --seed $seed --model-path $model_path --comm-model $comm_model --K-neighbor $K_neighbor
#CUDA_VISIBLE_DEVICES=0 python inference_by_dronet.py $arch --vinit $vint  --radius $radius --K $filterLength --F $n_vis --seed $seed --model-path $model_path --comm-model $comm_model --K-neighbor $K_neighbor

#seed=3
#CUDA_VISIBLE_DEVICES=0 python inference_by_dronet.py $arch --vinit $vint  --radius $radius --K $filterLength --F $n_vis --seed $seed --model-path $model_path --comm-model $comm_model --K-neighbor $K_neighbor
#CUDA_VISIBLE_DEVICES=0 python inference_by_dronet.py $arch --vinit $vint  --radius $radius --K $filterLength --F $n_vis --seed $seed --model-path $model_path --comm-model $comm_model --K-neighbor $K_neighbor
#CUDA_VISIBLE_DEVICES=0 python inference_by_dronet.py $arch --vinit $vint  --radius $radius --K $filterLength --F $n_vis --seed $seed --model-path $model_path --comm-model $comm_model --K-neighbor $K_neighbor


