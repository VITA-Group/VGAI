from problems.problem import Problem
import numpy as np
import scipy.linalg
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels
import torch


class LQRProblem(Problem):
    def __init__(self, config):
        """
        Initialize the LQRProblem object
        Args:
            config (): a dict with the parameters for the experiment
        """
        super(LQRProblem, self).__init__(config)

        self.n_nodes = int(config['network_size'])
        self.dt = float(config['sampling_dt'])
        # self.sparse_system = config.getboolean('sparse_system')
        # self.symmetric_network = config.getboolean('symmetric_network')
        self.x_range = 7.0
        self.x_max = float(config['xmax'])
        self.var = float(config['system_variance'])
        # self.std_dev = np.sqrt(self.var * self.dt)
        # self.graph_type = config['graph_type']
        self.degree = int(config['degree'])
        self.b_scale = float(config['b_scale'])
        self.latex_print = False

        self.alpha = float(config['alpha'])

        if 'steady_state_cost' in config:
            self.steady_state_cost = int(config['steady_state_cost'])
        else:
            self.steady_state_cost = 0

        # initialize graph
        # a_net = np.zeros((self.n_nodes, self.n_nodes))
        # a_sys = np.zeros((self.n_nodes, self.n_nodes))
        # min_degree = 0
        # det = 0
        # max_degree = np.inf
        # while min_degree == 0 or np.abs(det) < 1e-6 or max_degree > 2 * self.degree:  # ensure non-singular and degree > 0
        #     # x_loc = np.random.rand(self.n_nodes, 2) * self.x_range - self.x_range / 2.0
        #     # generate node locations
        #     node_loc = np.random.uniform(-self.x_range, self.x_range, size=(self.n_nodes, 2))
        #
        #     # generate linear system and geometric network
        #     a_sys = pairwise_kernels(node_loc, metric='rbf')
        #     np.fill_diagonal(a_sys, 0)
        #     neigh = NearestNeighbors(n_neighbors=self.degree)
        #     neigh.fit(node_loc)
        #     a_net = a_sys * np.array(neigh.kneighbors_graph(mode='connectivity').todense())
        #
        #     # if self.graph_type == 'er':  # erdos renyi graph
        #     #     prob = self.degree / self.n_nodes
        #     #     a = np.random.binomial(1, prob, self.n_nodes * self.n_nodes).reshape((self.n_nodes, self.n_nodes))
        #     #     a_net = np.tril(a) + np.tril(a, -1).T
        #     #     np.fill_diagonal(a_net, 0)
        #     #
        #     # else:  # graph type is 'geometric'
        #     #     # generate agent locations
        #     #     x_loc = np.random.rand(self.n_nodes, 2) * self.x_range - self.x_range / 2.0
        #     #
        #     #     # get degree-nearest neighbors graph
        #     #     neigh = NearestNeighbors(n_neighbors=self.degree)
        #     #     neigh.fit(x_loc)
        #     #     a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())
        #     #     np.fill_diagonal(a_net, 0)
        #     #
        #     #     # make network symmetric
        #     #     if self.symmetric_network:
        #     #         # need degree >= self.degree
        #     #         a_net = a_net + a_net.T
        #     #         a_net[a_net > 0] = 1.0
        #
        #     det = np.linalg.det(a_net)
        #     min_degree = np.min(np.sum(a_net, axis=1))
        #     max_degree = np.max(np.sum(a_net, axis=1))

        # generate node locations
        node_loc = self.alpha * np.random.uniform(0, 1.0, size=(self.n_nodes, 2))

        # generate linear system and geometric network
        a_sys = pairwise_kernels(node_loc, metric='rbf')
        np.fill_diagonal(a_sys, 0)
        neigh = NearestNeighbors(n_neighbors=self.degree)
        neigh.fit(node_loc)

        a_net = a_sys * np.array(neigh.kneighbors_graph(mode='connectivity').todense())

        # discretize system given dt
        a_net = a_net / max(np.abs(np.linalg.eigvals(a_net)))

        a_expm = scipy.linalg.expm(self.dt * a_sys)
        b_sys = (np.linalg.inv(a_sys).dot(a_expm - np.eye(self.n_nodes))).dot(self.b_scale * np.eye(self.n_nodes))

        # simplified since A is symmetric:
        q_sys = (np.linalg.inv(2 * a_sys).dot(scipy.linalg.expm(self.dt * 2.0 * a_sys) - np.eye(self.n_nodes)))
        # q_sys is ALMOST symmetric within 1e-16
        q_sys = (q_sys + q_sys.T) / 2.0

        # a_sys = np.eye(self.n_nodes) + a_sys * self.dt

        self.a_net = a_net
        self.a_sys = a_expm
        self.b_sys = b_sys
        self.q_sys = q_sys
        self.r_sys = self.dt * np.eye(self.n_nodes) * (self.b_scale ** 2)  # TODO describe in paper
        self.cov = q_sys * self.var
        # self.mean = np.zeros((self.n_nodes,))
        self.std_dev = np.sqrt(self.cov[0, 0])

        self.k = self.dlqr()

        self.a_net_nan = self.a_net.reshape((self.n_nodes, self.n_nodes, 1))
        self.a_net_nan[self.a_net_nan == 0] = np.nan

    def initialize(self):
        """
        Randomly initialize the state of the system

        Returns: the initial state

        """
        return np.random.uniform(low=-self.x_max, high=self.x_max, size=(self.n_nodes,))

    def forward(self, xt, ut):
        """
        Computes the forward model for the system

        Args:
            xt (): current state of the system
            ut (): control action

        Returns: next state computed according to the discrete LQR solution

        """

        xt.shape = (self.n_nodes, 1)
        ut.shape = (self.n_nodes, 1)
        xt1 = self.a_sys.dot(xt) + self.b_sys.dot(ut) + np.random.normal(0, self.std_dev, (self.n_nodes, 1))
        # xt1 = (self.a_sys.dot(xt) + self.b_sys.dot(ut)).flatten() \
        #       + np.random.multivariate_normal(self.mean, self.cov).flatten()
        return xt1

    def get_aggregated_trajectory(self):
        """
        Collects an aggregation GNN dataset with n_episodes

        Returns:
            z : aggregated states from all nodes, all time points, shuffled
            u : optimal control actions corresponding to each aggregated state

        """

        n_data = self.episode_len * self.n_nodes * self.n_episodes
        z = np.zeros((n_data, self.filter_len))
        u = np.zeros((n_data, 1))
        x_agg = np.zeros((self.n_nodes, self.filter_len))
        ind = 0

        for i in range(0, self.n_episodes):

            x_agg.fill(0)
            xt = self.initialize()

            for t in range(0, self.episode_len ):
                x_agg = self.aggregate(x_agg, xt)
                ut = self.controller(xt)
                xt = self.forward(xt, ut)

                # train for all nodes' data
                z[ind:(ind + self.n_nodes), :] = x_agg  # .reshape((self.n_nodes, self.filter_len))
                u[ind:(ind + self.n_nodes), :] = ut.reshape(self.n_nodes, 1)
                ind = ind + self.n_nodes

        # shuffle data across trajectories, time instances, and node locations
        s = np.arange(z.shape[0])
        np.random.shuffle(s)
        z = z[s, :]
        u = u[s]

        return z, u

    def controller(self, x):
        """
        Computes the optimal controller via the LQR solution
        Args:
            x (): a state of the system

        Returns: the optimal control action for the system

        """
        x.shape = (self.n_nodes, 1)
        u = -self.k.dot(x)
        return u

    def dlqr(self):
        """
        Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        http://www.mwm.im/lqr-controllers-with-python/

        Returns:
            K : the optimal linear controller
        """
        x = np.matrix(scipy.linalg.solve_discrete_are(self.a_sys, self.b_sys, self.q_sys, self.r_sys))
        k = np.matrix(scipy.linalg.inv(self.b_sys.T * x * self.b_sys + self.r_sys) * (self.b_sys.T * x * self.a_sys))
        return k

    def instant_cost(self, xt, ut):
        """
        Compute the cost of this state and action according to the Q and R matrices

        Args:
            xt (): state
            ut (): action

        Returns: instantaneous LQR cost for a state and action

        """
        xt.shape = (self.n_nodes, 1)
        ut.shape = (self.n_nodes, 1)
        cost = xt.T.dot(self.q_sys).dot(xt) + ut.T.dot(self.r_sys).dot(ut)
        return cost

    def aggregate(self, x_agg, xt):
        """
        Perform aggegration operation
        Args:
            x_agg (): Last time step's aggregated info
            xt (): Current state of all agents

        Returns:
            Aggregated state values
        """
        # get rid of oldest forwarded information
        last_agg = np.array(x_agg[:, :-1]).reshape((self.n_nodes, 1, self.filter_len - 1))

        # get forwarded information from neighbors
        features = np.nansum(last_agg * self.a_net_nan, axis=0).reshape((self.n_nodes, self.filter_len - 1))
        return np.hstack((xt.reshape(self.n_nodes, 1), features))

    def get_cost_expert(self, seed_state=None):
        """
        Compute the median cost over num_test_traj trajectories of the expert controller
        Args:
            seed_state (): seed_state for random number generator for reproducibility of results

        Returns: median cost of a trajectory

        """
        if seed_state is not None:
            np.random.set_state(seed_state)

        costs = np.zeros((self.num_test_traj,))
        for n in range(0, self.num_test_traj):

            xt = self.initialize()
            for t in range(0, self.episode_len - 1):
                ut = self.controller(xt)
                if t >= self.steady_state_cost:
                    costs[n] = costs[n] + self.instant_cost(xt, ut)  # * self.dt
                xt = self.forward(xt, ut)

        return costs  #np.mean(costs), np.std(costs)

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
        x_agg_t = np.zeros((self.n_nodes, self.filter_len))

        for n in range(0, self.num_test_traj):

            # initialize trajectory
            x_t = self.initialize()
            x_agg_t.fill(0)

            # run the controller
            for t in range(0, self.episode_len - 1):
                # aggregation
                x_agg_t = self.aggregate(x_agg_t, x_t)
                x_agg_torch = x_agg_t.reshape((self.n_nodes, self.filter_len))
                x_agg_torch = torch.from_numpy(x_agg_torch).to(device=device)

                # compute controller
                u_t = model(x_agg_torch).data.cpu().numpy().reshape(self.n_nodes, 1)
                if t >= self.steady_state_cost:
                    costs[n] = costs[n] + self.instant_cost(x_t, u_t)  # * self.dt
                x_t = self.forward(x_t, u_t)

        return costs #np.mean(costs), np.std(costs)

    def print_costs(self, section, costs):
        """
        Pretty print table of costs
        Args:
            section (): string for section name
            d_costs (): dictionary of cost stats (mean, std dev) to print
        """
        expert = np.concatenate([cost['expert'] for cost in costs])
        nn = np.concatenate([cost['nn'] for cost in costs])

        d_costs = {'expert_mean': np.mean(expert), 'expert_std': np.std(expert), 'nn_mean': np.mean(nn), 'nn_std': np.std(nn)}
        if self.latex_print:
            print(
                section + ' & ' + ('%.2E' % d_costs['expert_mean']) + ' $\\pm$ ' + (
                        '%.2E' % d_costs['expert_std']) + ' & ' + ('%.2E' % d_costs['nn_mean']) + ' $\\pm$ ' + (
                        '%.2E' % d_costs['nn_std']) + ' \\\\')
        else:

            print(
                section + '\t' + ('%.2E' % d_costs['expert_mean']) + '\t' + (
                        '%.2E' % d_costs['expert_std']) + '\t' + ('%.2E' % d_costs['nn_mean']) + '\t' + (
                        '%.2E' % d_costs['nn_std']))

    def print_header(self, header):
        """
        Print header for table
        Args:
            header (): header to print
        """
        if self.latex_print:
            print(header + ' & Optimal Cost & NN Cost \\\\')
        else:
            print(header + '	OptimalMean	OptimalStd	NNMean	NNStd')

    def get_stats(self, model, device):
        """
        Test the LQR solution and trained GNN model to get mean cost and std dev
        Args:
            seed_state (): random seed state for reproducibility
            model (): trained PyTorch model
            device (): PyTorch device (GPU or CPU)

        Returns: dictionary of cost stats (mean, std dev) to print

        """
        expert = self.get_cost_expert()
        nn = self.get_cost_model(model, device)
        return {'expert': expert, 'nn': nn}

        # expert_mean, expert_std = self.get_cost_expert()
        # nn_mean, nn_std = self.get_cost_model(model, device)
        #
        # return {'expert_mean': expert_mean, 'expert_std': expert_std, 'nn_mean': nn_mean, 'nn_std': nn_std}
