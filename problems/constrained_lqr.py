from problems.lqr import LQRProblem
import numpy as np
import cvxpy as cvx


class ConstrainedLQRProblem(LQRProblem):
    def __init__(self, config):
        """
        Initialize the Constrained LQR problem by loading the constraints
        Args:
            config (): dict containing experiment parameters
        """
        super(ConstrainedLQRProblem, self).__init__(config)
        self.u_min = np.ones((self.n_nodes,)) * float(self.config['u_min'])
        self.u_max = np.ones((self.n_nodes,)) * float(self.config['u_max'])
        self.lookahead = int(self.config['lookahead'])

    def controller(self, x):
        """
        Computes the optimal controller via MPC
        Code from http://www.philipzucker.com/lqr-with-cvxpy/

        Args:
            x (): state of the system

        Returns: the optimal control action for the system

        """
        x.shape = (self.n_nodes, 1)
        xt = cvx.Variable(self.n_nodes)
        state = [xt]
        cost = 0
        constraints = [xt == x]
        controls = []

        for i in range(self.lookahead):
            ut = cvx.Variable(self.n_nodes)
            xtn = cvx.Variable(self.n_nodes)
            controls.append(ut)
            state.append(xtn)

            # add contraints
            constraints.append(xtn == self.a_sys * xt + self.b_sys * ut)
            constraints.append(ut <= self.u_max)
            constraints.append(ut >= self.u_min)

            cost = cost + cvx.sum_squares(xtn) + cvx.sum_squares(ut)  # TODO assume q and r are identity
            xt = xtn

        objective = cvx.Minimize(cost)
        prob = cvx.Problem(objective, constraints)
        sol = prob.solve(verbose=False)
        u = np.array(list(map(lambda x: x.value, controls)))

        return u[0, :]



