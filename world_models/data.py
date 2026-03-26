import pybullet as p
import pybullet_data
import numpy as np
import random
import pickle
import time

class PandaEnv:
    def __init__(self, gui=True):
        self.gui = gui
        self.connect()

        self.ee_link = 11
        self.arm_joints = list(range(7))
        self.gripper_joints = [9, 10]

        self.box_size = 0.035

        self.reset()

    # =========================
    # 🔌 连接
    # =========================
    def connect(self):
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(
            numSolverIterations=200,
            fixedTimeStep=1./240.
        )

    # =========================
    # 🔄 重置环境
    # =========================
    def reset(self):
        p.resetSimulation()

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            [0, 0, 0],
            useFixedBase=True
        )

        p.loadURDF("plane.urdf")

        self.reset_robot()

        # 随机一个箱子（简化）
        x = random.uniform(0.4, 0.6)
        y = random.uniform(-0.2, 0.2)
        z = self.box_size

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.box_size]*3
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.box_size]*3,
            rgbaColor=[0.8,0.6,0.4,1]
        )

        self.box = p.createMultiBody(0.5, col, vis, [x,y,z])

        for _ in range(50):
            p.stepSimulation()

    # =========================
    # 🤖 初始姿态
    # =========================
    def reset_robot(self):
        poses = [0,-0.4,0,-2.5,0,2.2,0.8,0,0,0.04,0.04]
        for i in range(len(poses)):
            p.resetJointState(self.robot_id, i, poses[i])

    # =========================
    # 📊 状态（非常关键）
    # =========================
    def get_state(self):
        # 关节
        joint_states = p.getJointStates(self.robot_id, self.arm_joints)
        q = np.array([s[0] for s in joint_states])
        dq = np.array([s[1] for s in joint_states])

        # 末端位置
        ee_pos = p.getLinkState(self.robot_id, self.ee_link)[0]

        # 箱子
        box_pos, _ = p.getBasePositionAndOrientation(self.box)

        state = np.concatenate([q, dq, ee_pos, box_pos])
        return state

    # =========================
    # 🎮 动作（核心）
    # =========================
    def step(self, action):
        """
        action: 7维 torque 或 joint target
        """
        for i, j in enumerate(self.arm_joints):
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.TORQUE_CONTROL,
                force=action[i]
            )

        p.stepSimulation()

        if self.gui:
            time.sleep(1./240.)

        return self.get_state()

    # =========================
    # 🎲 随机动作
    # =========================
    def sample_action(self):
        return np.random.uniform(-5, 5, size=7)


# =========================
# 📦 数据采集主程序
# =========================
def collect_data(num_steps=5000, save_path="panda_data.pkl"):
    env = PandaEnv(gui=False)

    data = []

    for step in range(num_steps):
        if step % 200 == 0:
            env.reset()

        state = env.get_state()
        action = env.sample_action()
        next_state = env.step(action)

        data.append((state, action, next_state))

        if step % 500 == 0:
            print(f"Collected {step} steps")

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print("✅ 数据采集完成:", len(data))


if __name__ == "__main__":
    collect_data()
