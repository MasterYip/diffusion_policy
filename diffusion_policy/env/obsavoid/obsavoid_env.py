
import numpy as np
from matplotlib import pyplot as plt

class Obstacle1dEnv(object):
    """Obstacle environment
    Simple 1d environment which has only 1 dimension.
    Feasible region are defined as $F:{x|x \in [lb1, ub1] U [lb2, ub2] ...}$

    Args:
        object (_type_): _description_
    """

    def __init__(self, y=0.0, v=0.0, env_step=0.01, vis=True):
        # state
        self.y = y
        self.v = v
        self.t = 0
        self.y_hist = []
        self.hist_len = 1000

        # Acc Scaling: scaling for acceleration action input
        self.acc_scale = 100
        # Vel Scaling: scaling for velocity observation output
        self.vel_scale = 10

        # PID
        self.p = 10000
        self.d = 200

        # config
        self.env_step = env_step
        self.bounds = []

        # vis
        self.vis = vis
        self.vis_time_window = 1
        self.bd_vis_sample = 100
        if self.vis:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.show(block=False)

    # Config
    def add_boundfunc(self, lb_fun, ub_fun):
        self.bounds.append((lb_fun, ub_fun))

    def step_env(self, acc=0.0):
        """Step env
        Args:
            acc (float, optional): Acc scaled. Defaults to 0.0.
        """
        self.t += self.env_step
        self.y += self.v * self.env_step
        self.v += acc * self.env_step * self.acc_scale
        self.y_hist.append(self.y)
        if len(self.y_hist) > self.hist_len:
            self.y_hist.pop(0)
        if self.vis:
            self.vis_step()

    def step_env_y(self, y):
        """Step env with y
        Args:
            y (float): y
        """
        self.t += self.env_step
        self.y = float(y)
        self.y_hist.append(self.y)
        if len(self.y_hist) > self.hist_len:
            self.y_hist.pop(0)
        if self.vis:
            self.vis_step()

    # Reference
    def sdf_value(self, y, t=None):
        values = []
        if not t:
            t = self.t
        for bd in self.bounds:
            if y > (bd[0](t)+bd[1](t))/2:
                values.append(bd[1](t) - y)
            else:
                values.append(y - bd[0](t))
        return max(values)

    def get_ref_target(self):
        """Heuristic target for PID
        TODO: Too simple
        """
        bd = self.bounds[0]
        return (bd[0](self.t)+bd[1](self.t))/2

    def pid_ctrl(self):
        target = self.get_ref_target()
        return self.p*(target-self.y) + self.d*(-self.v)

    # Dataset
    def get_observation(self, n=6, step=None):
        """Observation is state(2) + sdf_obs
        """
        if step is None:
            step = self.env_step
        state = [self.y, self.v/self.vel_scale]
        # next n sdf_value
        sdf_obs = [self.sdf_value(self.y, self.t + i*step) for i in range(n)]
        return state + sdf_obs

    def get_action(self):
        return [self.pid_ctrl()/self.acc_scale]
    
    def get_noised_action(self, noise=1.0):
        return [self.pid_ctrl()/self.acc_scale + np.random.randn()*noise]

    def get_reward(self):
        # return [self.sdf_value(self.y)]
        return self.sdf_value(self.y)

    # Vis
    def vis_step(self):
        self.ax.clear()
        self.ax.plot(np.linspace(self.t-self.env_step*len(self.y_hist), self.t, len(self.y_hist)), self.y_hist)
        X = np.linspace(self.t-self.vis_time_window, self.t+self.vis_time_window, self.bd_vis_sample)
        for bd in self.bounds:
            self.ax.plot(X, bd[0](X), 'r')
            self.ax.plot(X, bd[1](X), 'g')
        self.ax.set_ylim(-4, 4)
        self.ax.set_xlim(self.t-self.vis_time_window, self.t+self.vis_time_window)
        plt.pause(0.00001)

    def vis_scatter(self, X, Y):
        # scatter marker size is 1
        self.ax.scatter(X, Y, s=1)
        plt.pause(0.00001)


def sine_bound_env(vis=True, y=0.0, v=0.0, env_step=0.01):
    env = Obstacle1dEnv(y=y, v=v, env_step=env_step, vis=vis)
    # env.add_boundfunc(lambda t: 0.5*np.sin(5*t)-0.2, lambda t: 0.5*np.sin(5*t)+0.2+0.1*np.sin(30*t))
    env.add_boundfunc(lambda t: 0.5*np.sin(15*t)-0.3, lambda t: 0.5*np.sin(15*t)+0.3)
    # env.add_boundfunc(lambda t: 0.5*np.sin(15*t)-0.2-1.5, lambda t: 0.5*np.sin(15*t)+0.2-1.5)
    return env


def increase_bound_env(vis=True, y=0, v=0, env_step=0.01):
    env = Obstacle1dEnv(y=y, v=v, env_step=env_step, vis=vis)
    env.add_boundfunc(lambda t: 0.3*t-0.2, lambda t: 0.3*t+0.2+0.1*np.sin(10*t))
    return env
