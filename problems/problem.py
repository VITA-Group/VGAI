from abc import ABC, abstractmethod
import torch
import os
import pickle

class Problem(ABC):

    def __init__(self, config):
        """
        Initialize the problem object
        Args:
            config (): dict which contains the parameters for any Problem, and then the specific Problem subtype
        """
        super(Problem, self).__init__()
        self.config = config
        self.one_model = config.getboolean('one_model')
        self.debug = config.getboolean('debug')
        self.delayed_x = config.getboolean('delayed_x')
        self.filter_len = int(config['filter_length'])
        self.episode_len = int(config['episode_len'])
        self.n_episodes = int(config['n_episodes'])
        self.num_test_traj = int(config['num_test_traj'])

    @abstractmethod
    def initialize(self):
        """
        Initialize the state for the system
        """
        pass

    @abstractmethod
    def get_cost_model(self, model, device, seed_state=None):
        pass

    @abstractmethod
    def print_costs(self, section, d_costs):
        pass

    @abstractmethod
    def print_header(self, header):
        pass

    @abstractmethod
    def get_stats(self, model, device):
        pass

    @abstractmethod
    def get_cost_expert(self, seed_state=None):
        pass

    @abstractmethod
    def get_aggregated_trajectory(self):
        """
        Returns a data set containing the aggregated state vectors and optimal output actions
        """
        pass

    @abstractmethod
    def forward(self, x, u):
        """
        The forward model for the system. Given:
        Args:
            x (): the current state
            u (): the control action

        Returns the next state
        """
        pass

    @abstractmethod
    def controller(self, x):
        """
        The optimal controller for the system
        Args:
            x (): the current state
        Returns: the optimal action
        """
        pass

    def save_model(self, my_section, model):
        """
        Save the PyTorch model to file
        Args:
            my_section (): string for the folder name of this experiment
            model (): PyTorch model to save
        """
        # make a folder
        if not os.path.exists(my_section):
            os.makedirs(my_section)

        # save problem object, and pytorch model
        torch.save(model.state_dict(), my_section + '/model.pth')
        pickle.dump(self, open(my_section + '/problem.pkl', 'wb'))

    def save_data(self, my_section, x_train, y_train):
        """
        Save the training data for reuse in later experiments
        Args:
            my_section (): string for the folder name of this experiment
            x_train (): aggregated state data
            y_train (): optimal action data
        """
        # make a folder
        if not os.path.exists(my_section):
            os.makedirs(my_section)

        # save the data
        pickle.dump(x_train, open(my_section + '/x_train.pkl', 'wb'))
        pickle.dump(y_train, open(my_section + '/y_train.pkl', 'wb'))

    def load_data(self, my_section):
        """
        Load data from file to be used in training
        Args:
            my_section (): string for the folder name of this experiment

        Returns:
            x_train : aggregated state data
            y_train : optimal action data

        """
        x_train = pickle.load(open(my_section + '/x_train.pkl', 'rb'))
        y_train = pickle.load(open(my_section + '/y_train.pkl', 'rb'))
        return x_train, y_train

    def get_average_cost(self, model, device):
        """
        Compute the average cost over num_test_traj trajectories for the expert controller and the trained model
        Args:
            model (): PyTorch model trained to mimic the optimal action
            device (): PyTorch device (GPU or CPU) on which the model runs

        Returns:
            avg_cost_nn : average cost accumulated by the learned controller
            avg_cost_u : average cost accumulated by the optimal controller

        """
        avg_cost_nn = 0
        avg_cost_u = 0
        for n in range(0, self.num_test_traj):
            _, _, c_nn, c_u = self.test_model(model, device)
            avg_cost_nn = avg_cost_nn + c_nn / float(self.num_test_traj)
            avg_cost_u = avg_cost_u + c_u / float(self.num_test_traj)
        return avg_cost_nn, avg_cost_u
