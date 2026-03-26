import pybullet as p
import pybullet_data
import time
import random
import numpy as np

class StablePandaTask:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(numSolverIterations=200, fixedTimeStep=1./240.)

        p.resetDebugVisualizerCamera(1.5, 50, -35, [0.3, 0, 0.2])

        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0,0,0], useFixedBase=True)
        p.loadURDF("plane.urdf")

        self.ee_link = 11
        self.arm_joints = list(range(7))
        self.gripper_joints = [9,10]

        self.box_size = 0.035
        self.gripper_open = 0.04
        self.gripper_close = 0.0

        self.safe_h = 0.55
        self.lift_h = 0.75

        self.target = [0.4, 0.4]

        self.reset_robot()
        self.boxes = self.spawn_boxes(3)
        self.obstacles = self.spawn_obstacles(3)

        for _ in range(120):
            p.stepSimulation()

    # =========================
    # 🤖 初始化姿态
    # =========================
    def reset_robot(self):
        poses = [0,-0.4,0,-2.5,0,2.2,0.8,0,0,0.04,0.04]
        for i in range(len(poses)):
            p.resetJointState(self.robot_id, i, poses[i])

    # =========================
    # 📦 箱子
    # =========================
    def spawn_boxes(self, n):
        objs = []
        for i in range(n):
            x = random.uniform(0.4,0.6)
            y = random.uniform(-0.2,0.2)
            z = self.box_size

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.box_size]*3)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.box_size]*3,
                                      rgbaColor=[0.8,0.6,0.4,1])

            box = p.createMultiBody(0.5, col, vis, [x,y,z])
            p.changeDynamics(box,-1,lateralFriction=1.0)

            objs.append(box)
        return objs

    # =========================
    # 🚧 障碍物
    # =========================
    def spawn_obstacles(self, n):
        obs = []
        for _ in range(n):
            x = random.uniform(0.4,0.6)
            y = random.uniform(-0.25,0.25)

            r = 0.05
            h = 0.25

            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                      rgbaColor=[0.2,0.2,0.8,1])

            body = p.createMultiBody(0, col, vis, [x,y,h/2])
            obs.append((body,r))
        return obs

    # =========================
    # 🧠 平滑运动
    # =========================
    def move_smooth(self, start, end, grip, steps=120):
        orn = p.getQuaternionFromEuler([0,np.pi,np.pi/2])

        for t in np.linspace(0,1,steps):
            pos = (1-t)*np.array(start) + t*np.array(end)

            joint = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link,
                pos.tolist(),
                orn
            )

            for i in self.arm_joints:
                p.setJointMotorControl2(self.robot_id, i,
                    p.POSITION_CONTROL, joint[i], force=300)

            for j in self.gripper_joints:
                p.setJointMotorControl2(self.robot_id, j,
                    p.POSITION_CONTROL, grip, force=200)

            p.stepSimulation()
            time.sleep(1./240.)

    # =========================
    # 🤏 抓取（关键）
    # =========================
    def grasp(self, box):
        cid = p.createConstraint(
            self.robot_id, self.ee_link,
            box, -1,
            p.JOINT_FIXED,
            [0,0,0],
            [0,0,0],
            [0,0,0]
        )
        return cid

    def release(self, cid):
        if cid:
            p.removeConstraint(cid)

    # =========================
    # 🚀 主流程
    # =========================
    def run(self):
        cur = [0,0,self.safe_h]

        for i, box in enumerate(self.boxes):
            pos,_ = p.getBasePositionAndOrientation(box)
            x,y,z = pos

            print(f"📦 处理箱子 {i+1}")

            # 1. 上方移动
            self.move_smooth(cur, [x,y,self.safe_h], self.gripper_open)
            cur = [x,y,self.safe_h]

            # 2. 下去（关键：+0.02 防穿透）
            self.move_smooth(cur, [x,y,z+0.02], self.gripper_open)
            cur = [x,y,z+0.02]

            # 3. 夹紧
            self.move_smooth(cur, cur, self.gripper_close, steps=40)

            # 4. 绑定（真正抓住）
            cid = self.grasp(box)

            # 5. 提起
            self.move_smooth(cur, [x,y,self.safe_h], self.gripper_close)
            cur = [x,y,self.safe_h]

            # ===== 避障三段式 =====
            self.move_smooth(cur, [x,y,self.lift_h], self.gripper_close)
            self.move_smooth([x,y,self.lift_h],
                             [self.target[0],self.target[1],self.lift_h],
                             self.gripper_close)
            self.move_smooth([self.target[0],self.target[1],self.lift_h],
                             [self.target[0],self.target[1],self.safe_h],
                             self.gripper_close)

            cur = [self.target[0],self.target[1],self.safe_h]

            # 6. 放下
            stack_z = self.box_size + i*(self.box_size*2)
            self.move_smooth(cur, [cur[0],cur[1],stack_z], self.gripper_close)
            cur = [cur[0],cur[1],stack_z]

            # 7. 释放
            self.release(cid)

            # 8. 抬起
            self.move_smooth(cur, [cur[0],cur[1],self.safe_h], self.gripper_open)
            cur = [cur[0],cur[1],self.safe_h]

        print("✅ 完成任务")

        while True:
            p.stepSimulation()
            time.sleep(0.01)


if __name__ == "__main__":
    env = StablePandaTask()
    env.run()
