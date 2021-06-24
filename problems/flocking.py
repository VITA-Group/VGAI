from problems.problem import Problem
import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
from scipy.stats import iqr
#np.random.seed(0)


class FlockingProblem(Problem):
    def __init__(self, config):
        """
        Initialize the flocking problem
        Args:
            config (): dict containing experiment parameters
        """
        super(FlockingProblem, self).__init__(config)

        self.nx_system = 4
        self.n_nodes = int(config['network_size'])
        self.comm_radius = float(config['comm_radius'])
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max  # 0.5 * self.v_max
        self.r_max = float(config['max_rad_init']) 
        self.std_dev = float(config['std_dev']) * self.dt
        self.controller_gain = 1.0

        self.pooling = []
        if config.getboolean('sum_pooling'):
            self.pooling.append(np.nansum)
        if config.getboolean('min_pooling'):
            self.pooling.append(np.nanmin)
        if config.getboolean('max_pooling'):
            self.pooling.append(np.nanmax)
        self.n_pools = len(self.pooling)
        self.centralized_controller = config.getboolean('centralized_controller')

        # number of features and outputs
        self.n_features = int(config['N_features'])
        self.nx = int(self.n_features / self.n_pools / self.filter_len)
        self.nu = int(config['N_outputs'])  # outputs

        self.n_data_points = self.episode_len * self.n_nodes * self.n_episodes

    def potential_grad(self, pos_diff, r2):
        """
        Computes the gradient of the potential function for flocking proposed in Turner 2003.
        Args:
            pos_diff (): difference in a component of position among all agents
            r2 (): distance squared between agents

        Returns: corresponding component of the gradient of the potential

        """
        # r2 = r2 + 1e-4 * np.ones((self.n_agents, self.n_agents))  # TODO - is this necessary?
        grad = -2.0 * np.divide(pos_diff, np.multiply(r2, r2)) + 2 * np.divide(pos_diff, r2) 
        grad = grad * 0.1
        grad[r2 > self.comm_radius] = 0
        return grad  

    def initialize(self, seed=0):
        """
        Initialize the state of the flock. Agents positions are initialized according to a uniform distirbution
        on a disk around (0,0). Velocities are chosen according to a uniform distribution in a range.
        Also enforces constraints on the initial degree of the network and that agents are not initially too close.

        Returns:  Initial state of the flock

        """
        np.random.seed(seed)
        x = np.zeros((self.n_nodes, self.nx_system))
        degree = 0
        min_dist = 0

        while degree < 2 or min_dist < 0.1:  #< 0.25:  # 0.25:  #0.5: #min_dist < 0.25:
            # randomly initialize the state of all agents
            length = np.sqrt(np.random.uniform(0, self.r_max, size=(self.n_nodes,)))
            angle = np.pi * np.random.uniform(0, 2, size=(self.n_nodes,))
            x[:, 0] = length * np.cos(angle)
            x[:, 1] = length * np.sin(angle)

            bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
            x[:, 2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_nodes,)) + bias[0]
            x[:, 3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_nodes,)) + bias[1]

            # compute distances between agents
            x_t_loc = x[:, 0:2]  # x,y location determines connectivity
            a_net = squareform(pdist(x_t_loc.reshape((self.n_nodes, 2)), 'euclidean'))

            # no self loops
            a_net = a_net + 2 * self.comm_radius * np.eye(self.n_nodes)

            # compute minimum distance between agents and degree of network
            min_dist = np.min(np.min(a_net))
            a_net = a_net < self.comm_radius
            degree = np.min(np.sum(a_net.astype(int), axis=1))
        return x

    def controller(self, x):
        """
        The controller for flocking from Turner 2003.
        Args:
            x (): the current state
            centralized (): a boolean flag indicating if each agent should have access to all others' state info
                            (a centralized controller)
        Returns: the optimal action
        """
        centralized = self.centralized_controller
        s_diff = x.reshape((self.n_nodes, 1, self.nx_system)) - x.reshape((1, self.n_nodes, self.nx_system))
        r2 = np.multiply(s_diff[:, :, 0], s_diff[:, :, 0]) + np.multiply(s_diff[:, :, 1], s_diff[:, :, 1]) + np.eye(
            self.n_nodes)
        p = np.dstack((s_diff, self.potential_grad(s_diff[:, :, 0], r2), self.potential_grad(s_diff[:, :, 1], r2)))
        if not centralized:
            p = self.get_comms(p, self.get_connectivity(x))
        p_sum = np.nansum(p, axis=1).reshape((self.n_nodes, self.nx_system + 2))
        return np.hstack(((- p_sum[:, 4] - p_sum[:, 2]).reshape((-1, 1)), (- p_sum[:, 3] - p_sum[:, 5]).reshape(-1, 1)))

    def forward(self, x, u):
        """
        The forward model for the system. Given:
        Args:
            x (): current state
            u (): control action

        Returns the next state for the flock
        """

        x_ = np.zeros((self.n_nodes, self.nx_system))
        # x position
        x_[:, 0] = x[:, 0] + x[:, 2] * self.dt
        # y position
        x_[:, 1] = x[:, 1] + x[:, 3] * self.dt
        # x velocity
        x_[:, 2] = x[:, 2] + 0.1 * self.controller_gain * u[:, 0] * self.dt + np.random.normal(0, self.std_dev,
                                                                                               (self.n_nodes,))
        # + self.dt * np.random.normal(0, self.std_dev, (self.n_agents,))
        # y velocity
        x_[:, 3] = x[:, 3] + 0.1 * self.controller_gain * u[:, 1] * self.dt + np.random.normal(0, self.std_dev,
                                                                                               (self.n_nodes,))
        # + self.dt * np.random.normal(0, self.std_dev,(self.n_agents,))
        return x_

    def get_connectivity(self, x):
        """
        Get the adjacency matrix of the network based on agent locations by computing pairwise distances using pdist
        Args:
            x (): current states of all agents

        Returns: adjacency matrix of network

        """
        x_t_loc = x[:, 0:2]  # x,y location determines connectivity
        a_net = squareform(pdist(x_t_loc.reshape((self.n_nodes, 2)), 'euclidean'))
        a_net = (a_net < self.comm_radius).astype(float)
        np.fill_diagonal(a_net, 0)
        return a_net


    def get_knn_connectivity(self, x, K=10):
        """
        Get the adjacency matrix of the network based on agent locations by computing pairwise distances using pdist
        Args:
            x (): current states of all agents
            K (): number of neighbors
                    
        Returns: adjacency matrix of network

        """
        x_t_loc = x[:, 0:2]  # x,y location determines connectivity
        a_net = squareform(pdist(x_t_loc.reshape((self.n_nodes, 2)), 'euclidean'))
        for r_idx in range(self.n_nodes):
            thredshold = sorted(a_net[r_idx, :])[K]
            for c_idx in range(self.n_nodes):
                if a_net[r_idx, c_idx] < thredshold:
                    a_net[r_idx, c_idx] = 1
                else:
                    a_net[r_idx, c_idx] = 0
        a_net = np.transpose(a_net) + a_net
        a_net = np.sign(a_net).astype(float)
        np.fill_diagonal(a_net, 0)
        return a_net


    def get_x_features(self, xt):
        """
        Compute the non-linear features necessary for implementing Turner 2003
        Args:
            xt (): current state of all agents

        Returns: matrix of features for each agent

        """

        diff = xt.reshape((self.n_nodes, 1, self.nx_system)) - xt.reshape((1, self.n_nodes, self.nx_system))
        r2 = np.multiply(diff[:, :, 0], diff[:, :, 0]) + np.multiply(diff[:, :, 1], diff[:, :, 1]) + np.eye(
            self.n_nodes)
        return np.dstack((diff[:, :, 2], np.divide(diff[:, :, 0], np.multiply(r2, r2)), np.divide(diff[:, :, 0], r2),
                          diff[:, :, 3], np.divide(diff[:, :, 1], np.multiply(r2, r2)), np.divide(diff[:, :, 1], r2)))

    def get_features(self, agg):
        """
        Matrix of
        Args:
            agg (): the aggregated matrix from the last time step

        Returns: matrix of aggregated features from all nodes at current time

        """
        return np.tile(agg[:, :-self.nx].reshape((self.n_nodes, 1, -1)), (1, self.n_nodes, 1))  # TODO check indexing

    def get_comms(self, mat, a_net):
        """
        Enforces that agents who are not connected in the network cannot observe each others' states
        Args:
            mat (): matrix of state information for the whole graph
            a_net (): adjacency matrix for flock network (weighted networks unsupported for now)

        Returns:
            mat (): sparse matrix with NaN values where agents can't communicate

        """
        a_net[a_net == 0] = np.nan
        return mat * a_net.reshape(self.n_nodes, self.n_nodes, 1)

    def get_pool(self, mat, func):
        """
        Perform pooling operations on the matrix of state information. The replacement of values with NaNs for agents who
        can't communicate must already be enforced.
        Args:
            mat (): matrix of state information
            func (): pooling function (np.nansum(), np.nanmin() or np.nanmax()). Must ignore NaNs.

        Returns:
            information pooled from neighbors for each agent

        """
        return func(mat, axis=1).reshape((self.n_nodes, self.n_features))  # TODO check this axis = 1

    def aggregate(self, x_agg, xt):
        """
        Perform aggegration operation for all possible pooling operations using helper functions get_pool and get_comms
        Args:
            x_agg (): Last time step's aggregated info
            xt (): Current state of all agents

        Returns:
            Aggregated state values
        """
        x_features = self.get_x_features(xt)
        a_net = self.get_connectivity(xt)
        for k in range(0, self.n_pools):
            comm_data = self.get_comms(np.dstack((x_features, self.get_features(x_agg[:, :, k]))), a_net)
            x_agg[:, :, k] = self.get_pool(comm_data, self.pooling[k])
        return x_agg

    def get_aggregated_trajectory(self):
        """
        Returns: a data set containing the aggregated state vectors and optimal output actions

        """
        # TODO cannot have a separate NN for each node

        z = np.zeros((self.episode_len * self.n_nodes * self.n_episodes, self.n_features))
        u = np.zeros((self.episode_len * self.n_nodes * self.n_episodes, self.nu))
        x_agg = np.zeros((self.n_nodes, self.nx * self.filter_len, self.n_pools))
        ind = 0
        for i in range(0, self.n_episodes):
            x_agg.fill(0)
            xt = self.initialize()

            for t in range(0, self.episode_len):
                x_agg = self.aggregate(x_agg, xt)
                ut = self.controller(xt)
                xt = self.forward(xt, ut)

                # train for all nodes' data
                z[ind:(ind + self.n_nodes), :] = x_agg.reshape((self.n_nodes, self.n_features))
                u[ind:(ind + self.n_nodes), :] = ut
                ind = ind + self.n_nodes

        # shuffle data across trajectories, time instances, and node locations
        s = np.arange(z.shape[0])
        np.random.shuffle(s)
        z = z[s, :]
        u = u[s]

        return z, u

    def instant_cost(self, x, u):  # sum of differences in angles
        """
        Compute the instantantaneous cost of the state of the flock
        Args:
            x (): current state of the flock
            u (): current action

        Returns: a float value for the instantaneous cost

        """
        cost_x = np.sum(np.var(x[:, 2:4], axis=0))
        # cost_u = 0.001 * np.sum(np.sum(np.multiply(u, u)))
        # print(str(cost_x) + " " + str(cost_u))
        return cost_x  # + cost_u  # variance of the velocities
        # return np.sum(np.sum(np.abs(self.controller(x, centralized=True)))) # sum of optimal actions among all agents
        # + np.sum(pdist((x[:, 0:2]).reshape(-1, 1)))  # distance among all agents

    # def test_model(self, model, device):
    #     """
    #     Test the trained model on one system trajectory
    #     Args:
    #         model (): PyTorch model trained to mimic the optimal action
    #         device (): PyTorch device (GPU or CPU) on which the model runs
    #
    #     Returns trajectories and costs
    #     """
    #     # initialize trajectory
    #     xt1 = xt2 = self.initialize()
    #     x1 = np.zeros((self.n_nodes * self.nx_system, self.episode_len))
    #     x2 = np.zeros((self.n_nodes * self.nx_system, self.episode_len))
    #     x_agg1 = np.zeros((self.n_nodes, self.nx * self.filter_len, self.n_pools))
    #     cost_nn = cost_u = 0.0
    #
    #     # step through the trajectory
    #     for t in range(0, self.episode_len - 1):
    #         x1[:, t] = xt1.flatten()
    #         x2[:, t] = xt2.flatten()
    #
    #         # aggregate and compute learned control action
    #         x_agg1 = self.aggregate(x_agg1, xt1)
    #         x_agg1_t = x_agg1.reshape((self.n_nodes, self.n_features))
    #         x_agg1_t = torch.from_numpy(x_agg1_t).to(device=device)
    #         ut1 = model(x_agg1_t).data.cpu().numpy().reshape(self.n_nodes, self.nu)
    #
    #         # optimal controller
    #         # ut2 = self.controller(xt2, self.centralized_controller)
    #         ut2 = self.controller(xt2)  # TODO
    #
    #         # sum costs
    #         cost_nn = cost_nn + self.instant_cost(xt1, ut1) * self.dt
    #         cost_u = cost_u + self.instant_cost(xt2, ut2) * self.dt
    #
    #         xt1 = self.forward(xt1, ut1)
    #         xt2 = self.forward(xt2, ut2)
    #
    #     return x1, x2, cost_nn, cost_u

    def get_cost_expert(self, seed_state=None):
        """
        Compute the median cost over num_test_traj trajectories of the expert controller
        Args:
            seed (): seed_state for random number generator for reproducibility of results
            centralized_controller (): boolean indicating whether each agent has access to all others' data

        Returns: median cost of a trajectory

        """
        if seed_state is not None:
            np.random.set_state(seed_state)

        costs = np.zeros((self.num_test_traj,))
        for n in range(0, self.num_test_traj):
            xt = self.initialize()
            for t in range(0, self.episode_len - 1):
                ut = self.controller(xt)
                costs[n] = costs[n] + self.instant_cost(xt, ut) * self.dt
                xt = self.forward(xt, ut)
        return costs  # np.median(costs), iqr(costs)/2.0

    def get_cost_model(self, model, device, seed_state=None):
        """
        Compute the median cost over num_test_traj trajectories of the trained model controller
        Args:
            model (): PyTorch model trained to mimic the optimal action
            device (): PyTorch device (GPU or CPU) on which the model runs
            seed_state (): seed_state for random number generator for reproducibility of results

        Returns: median cost of a trajectory

        """
        if seed_state is not None:
            np.random.set_state(seed_state)
        costs = np.zeros((self.num_test_traj,))
        x_agg_t = np.zeros((self.n_nodes, self.nx * self.filter_len, self.n_pools))
        for n in range(0, self.num_test_traj):
            # initialize trajectory
            x_t = self.initialize()
            x_agg_t.fill(0)

            # run the controller
            for t in range(0, self.episode_len - 1):
                # aggregation
                x_agg_t = self.aggregate(x_agg_t, x_t)
                x_agg_torch = x_agg_t.reshape((self.n_nodes, self.n_features))
                x_agg_torch = torch.from_numpy(x_agg_torch).to(device=device)
                # compute controller
                u_t = model(x_agg_torch).data.cpu().numpy().reshape(self.n_nodes, self.nu)
                costs[n] = costs[n] + self.instant_cost(x_t, u_t) * self.dt
                x_t = self.forward(x_t, u_t)
        return costs  # np.median(costs), iqr(costs)/2.0

    def print_costs(self, section, costs):
        """
        Pretty print costs table
        Args:
            section (): string for section name
            d_costs (): dictionary of cost stats (mean, std dev) to print
        """

        d = np.concatenate([cost['d'] for cost in costs])
        c = np.concatenate([cost['c'] for cost in costs])
        nn = np.concatenate([cost['nn'] for cost in costs])

        c_med = np.median(c)
        c_iqr = iqr(c)/2.0
        d_med = np.median(d)
        d_iqr = iqr(d)/2.0
        nn_med = np.median(nn)
        nn_iqr = iqr(nn)/2.0

        print(section + ' & ' + ('%.2E' % c_med) + ' $\\pm$ ' + ('%.2E' % c_iqr) + ' & ' + (
                '%.2E' % d_med) + ' $\\pm$ ' + ('%.2E' % d_iqr) + ' & ' + (
                      '%.2E' % nn_med) + ' $\\pm$ ' + ('%.2E' % nn_iqr) + ' \\\\')

    def print_header(self, header):
        """
        Print header for table
        Args:
            header (): header to print
        """
        print(header + ' & Global Cost & Local Cost & NN Cost \\\\')

    def get_stats(self, model, device):
        """
        Test centralized, decentralized and GNN controllers to get mean cost and std dev
        Args:
            seed_state (): random seed state for reproducibility
            model (): trained PyTorch model
            device (): PyTorch device (GPU or CPU)

        Returns: dictionary of cost stats (mean, std dev) to print

        """
        # self.centralized_controller = True
        # c_mean, c_std = self.get_cost_expert()
        # self.centralized_controller = False
        # d_mean, d_std = self.get_cost_expert()
        # self.centralized_controller = True
        # nn_mean, nn_std = self.get_cost_model(model, device)
        # return {'c_mean': c_mean, 'c_std': c_std, 'd_mean': d_mean, 'd_std': d_std, 'nn_mean': nn_mean,
        #         'nn_std': nn_std}

        self.centralized_controller = True
        c = self.get_cost_expert()
        self.centralized_controller = False
        d = self.get_cost_expert()
        self.centralized_controller = True
        nn = self.get_cost_model(model, device)
        return {'c': c, 'd': d, 'nn': nn}
