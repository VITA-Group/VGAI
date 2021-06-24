from problems.lqr import LQRProblem
from problems.flocking import FlockingProblem
from problems.constrained_lqr import ConstrainedLQRProblem


def make_problem(config):
    """

    Args:
        config (): dict containing problem type

    Returns: initialized problem

    """
    problem_type = config['problem']
    if problem_type == 'LQRProblem':
        problem = LQRProblem(config)
    elif problem_type == 'ConstrainedLQRProblem':
        problem = ConstrainedLQRProblem(config)
    elif problem_type == 'FlockingProblem':
        problem = FlockingProblem(config)
    else:
        problem = None
    return problem