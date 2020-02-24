import matplotlib
# matplotlib.use('nbagg')
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np

class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval

    def append(self,obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel("X",fontsize=10)
        ax.set_ylabel("Y",fontsize=10)

        elems = []

        if self.debug:
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                     frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        if i >= (self.objects[0].trajectory[0].num):
            return

        self.objects[0].trajectory[0].selected_id = i
        ppc = PurePursuitControl(self.objects[0].trajectory[0])
        liner = ppc.getLiner(robot1.pose, i)
        omega = ppc.getOmega(robot1.pose, i)
        robot1.agent = Agent(liner, omega)

        while elems: elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)

class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):    # 引数を追加
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor
    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x,xn], [y,yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0),
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)),
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )

    def one_step(self, time_interval):
        if not self.agent: return
        obs =self.sensor.data(self.pose) if self.sensor else None #追加
        nu, omega = self.agent.decision(obs) #引数追加
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        if self.sensor: self.sensor.data(self.pose)

class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega

class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))

class Trajectory:
    def __init__(self, x, y):
        tmp_num = 0
        tmp_pos = list(range(0))
        for i in range(len(x)):
            tmp_pos.append([x[i], y[i]])
            tmp_num += 1

        self.pos = np.array(tmp_pos)
        self.num = tmp_num
        self.id = None
        self.selected_id = 0;

    def draw(self, ax, elems):
        for i in range(self.num):
            if i == self.selected_id:
                c = ax.scatter(self.pos[i][0], self.pos[i][1], s=10, label="trajectory", color="red")
                elems.append(c)
            else:
                c = ax.scatter(self.pos[i][0], self.pos[i][1], s=1, label="trajectory", color="green")
                elems.append(c)

class Map:
    def __init__(self):
        self.landmarks = []
        self.trajectory = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def append_trajectory(self, trajectory):
        trajectory.id = len(self.trajectory)
        self.trajectory.append(trajectory)

    def draw(self, ax, elems):
        for lm in self.landmarks: lm.draw(ax, elems)
        for tr in self.trajectory: tr.draw(ax, elems)

class IdealCamera:
    def __init__(self, env_map, \
                 distance_range=(0.5, 6.0),
                 direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos):
        if polarpos is None:
            return False

        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] \
                and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array( [np.hypot(*diff), phi ] ).T

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x,lx], [y,ly], color="pink")

class PurePursuitControl:
    def __init__(self, trajectory):
            self.trajectory = trajectory

    def getLiner(self, pose, ind):
        tx = self.trajectory.pos[ind][0]
        ty = self.trajectory.pos[ind][1]
        # 目標点までの距離を算出
        L = math.sqrt((tx - pose[0])**2 + (tx - pose[0])**2)
        Kp = 5
        liner = Kp * L
        return liner

    def getOmega(self, pose, ind):
        tx = self.trajectory.pos[ind][0]
        ty = self.trajectory.pos[ind][1]

        # 方位偏差
        error_angle = math.atan2(ty - pose[1], tx - pose[0]) - pose[2]

        # 目標点までの距離を算出
        L = math.sqrt((tx - pose[0])**2 + (tx - pose[0])**2)

        # 角速度指令値を算出
        omega = 0
        if L != 0:
            omega = 2.0 * 0.3 * math.sin(error_angle) / L

        return omega*5

if __name__ == '__main__':   ###name_indent
    world = World(30, 0.1)

    ### 地図を生成して3つランドマークを追加 ###
    m = Map()
    # 経路生成
    t = np.arange(0, 3.14*2, 0.1)
    cx = [3 * math.cos(ix) for ix in t]
    cy = [2 * math.sin(ix) for ix in t]
    m.append_trajectory(Trajectory(cx,cy))
    world.append(m)

    ### ロボットを作る ###
    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0/180*math.pi)
    robot1 = IdealRobot(np.array([3, 0, 0.7]).T, sensor=IdealCamera(m), agent=circling)

    world.append(robot1)

    ### アニメーション実行 ###
    world.draw()
