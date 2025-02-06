
from typing import overload
import numpy as np
import time
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

        # Default Obs Point Param
        self.obs_pt_param = [6, 5, 0.05, 0.3]

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
    def get_obspts(self):
        nt, ny, step, stepy = self.obs_pt_param
        if step is None:
            step = self.env_step
        obspts = []
        for i in range(ny):
            obspts += [[self.y + ((ny-1)/2-i)*stepy, self.t + j*step] for j in range(nt)]
        return obspts
    
    
    def get_observation(self):
        """Observation is state(2) + sdf_obs
        """
        state = [self.y, self.v/self.vel_scale]
        
        # next n sdf_value
        # sdf_obs = [self.sdf_value(self.y, self.t + i*step) for i in range(n)]
        sdf_obs = [self.sdf_value(y, t) for y, t in self.get_obspts()]
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

    def vis_scatter(self, T, Y):
        # scatter marker size is 1
        self.ax.scatter(T, Y, s=1)
        plt.pause(0.00001)

    def end(self):
        if self.vis:
            plt.close()

def sine_bound_env(vis=True, y=0.0, v=0.0, env_step=0.01):
    env = Obstacle1dEnv(y=y, v=v, env_step=env_step, vis=vis)
    # env.add_boundfunc(lambda t: 0.5*np.sin(5*t)-0.2, lambda t: 0.5*np.sin(5*t)+0.2+0.1*np.sin(30*t))
    env.add_boundfunc(lambda t: 0.5*np.sin(15*t)-0.3, lambda t: 0.5*np.sin(15*t)+0.3)
    # env.add_boundfunc(lambda t: 0.5*np.sin(15*t)-0.2-1.5, lambda t: 0.5*np.sin(15*t)+0.2-1.5)
    return env

def increase_bound_env(vis=True, y=0, v=0, env_step=0.01):
    env = Obstacle1dEnv(y=y, v=v, env_step=env_step, vis=vis)
    slope = 1.0
    env.add_boundfunc(lambda t: slope*t-0.2, lambda t: slope*t+0.2+0.1*np.sin(10*t))
    return env

def randpath_bound_env(vis=True, y=0, v=0, env_step=0.01):
    env = Obstacle1dEnv(y=y, v=v, env_step=env_step, vis=vis)
    slope_abs_bd = 1.0
    coef_abs_bd = [1.0, 1.0, 0.5, 0.5, 0.3, 0.2,0,0,0,0,0,0,0,0,0.1,0.1]
    width_bd = [0.6, 1.5]
    slope = (np.random.rand()-0.5)*2*slope_abs_bd
    coef = [(np.random.rand()-0.5)*2*coef_abs_bd[i//2] for i in range(len(coef_abs_bd)*2)]
    width = np.random.rand()*(width_bd[1]-width_bd[0])+width_bd[0]
    def func(t):
        res = slope*t
        for i in range(len(coef_abs_bd)):
            res += coef[i*2]*np.sin((i+1)*t) + coef[i*2+1]*np.cos((i+1)*t)
        return res
    env.add_boundfunc(lambda t: func(t)-width/2, lambda t: func(t)+width/2)
    return env

def test_rand_bound_env():
    env = randpath_bound_env()
    # stop when ctrl+c
    while True:
        num = 30
        fake_noise_scatter_X = np.linspace(env.t, env.t+0.3, num)
        fake_noise_scatter_Y = env.y + np.random.randn(num)*0.1
        env.step_env(acc=env.get_noised_action()[0])
        # env.vis_scatter(fake_noise_scatter_X, fake_noise_scatter_Y)
        pts = env.get_obspts()
        T = [pt[1] for pt in pts]
        Y = [pt[0] for pt in pts]
        env.vis_scatter(T, Y)
        # print(env.sdf_value(env.y))
        time.sleep(env.env_step)
        
if __name__ == "__main__":
    test_rand_bound_env()