The code is used to reproduce the result of "Scalable Perception-Action-Communication Loopswith Convolutional and Graph Neural Networks"
# Prerequisite
* Before running the script, make sure you have install microsoft airsim on 4.23 version. The instruction for the installation is in https://microsoft.github.io/AirSim/
* After installing airsim, launching the airsim by GUI and play start. Make sure the setting.json is correctly located in the airsim folder.
* Pytorch 1.0.0
* If you would like to get the environment we use for airsim, please replace the project file with ours in airsim folder.

# Quick start
* Run the batch file "exec_dagnn.sh"/"exec_grnn.sh"" for the training.
* replace the path of 'root_path' to change the path to store the dataset.
* change the setting according to the "Training hyper-parameters"

# Training hyper-parameters
* n_times : how many steps for each trajectory.
* n_agents : number of agents for the group.
* n_exp : number of training data for the initial size of dataset.
* start_idx : set 0 for the starting point of each trajectory.
* filterLength : the temporal length for DAGNN/GRNN.
* n_vis : number of feature dimension for transmission.
* radius : radius for the disk model.
* vinit : the maximum velocity for initialization.
* mode : 'optimal' using centralized controller to collect ground truth. 'vis_grnn'/'vis_dagnn'/ using other controllers to collect ground truth. 
* arch: 'vis_grnn'/'vis_dagnn' for the choice of archetecture.
* comm_model : 'disk' for disk model. 'knn' for knn model.
* n_exp_aug : number of datasize for the augmentation of dataset.


# Quick test
* Run the batch "exec_dagnn.sh"/"exec_gnn.sh"
* change the setting according to the "Testing hyper-parameters"

# Testing hyper-parameters
* n_agents : number of agents for each trajectory
* arch : 'vis_grnn'/'vis_dagnn' for the choice of archetecture.
* seed : the random seed for the initialization.
* Other hyper-parameters in "Training hyper-parameters".
