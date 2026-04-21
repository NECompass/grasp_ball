import numpy as np
import os
import mujoco
from scipy.spatial.transform import Rotation as R
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from stable_baselines3.common.env_checker import check_env

class PandaReachEnv(MujocoEnv, utils.EzPickle):  # 继承自Gymnasium官方的Mujoco包装类，处理底层的仿真步进；允许环境被序列化(保存为文件或在多进程中传输)
    metadata = {
        "render_modes":["human", "rgb_array"],
        "render_fps": 50
    }

    def __init__(self, **kwargs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        xml_path = os.path.join(parent_dir, 'grasp_task_rl.xml')
        frame_skip = 10  # 控制频率与物理频率的比例，物理引擎每走10步，算法才执行一次动作
        observation_space = Box(low=-np.inf, high = np.inf, shape = (32,), dtype = np.float64)
        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space = observation_space,
            **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)
        self.action_space = Box(low = -1, high = 1, shape = (8,), dtype = np.float64)
        self.joint_limits_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0])
        self.joint_limits_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04])

        self.table_id = self.model.geom("table_geom").id
        self.ball_geom_id = self.model.geom("ball_geom").id
        self.ball_joint_id = self.model.joint("ball_joint").id
        self.hand_id = self.model.geom("hand_collision").id
        self.left_site_id = self.model.site("left_finger_site").id
        self.right_site_id = self.model.site("right_finger_site").id
        self.robot_geom_ids = []
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if any(part in name for part in ["link", "hand", "finger"]):
                self.robot_geom_ids.append(i)

        # self.init_ball_pos = self.data.xpos[self.model.body("ball").id].copy()   # 这个值是[0,0,0]
        self.init_ball_pos = self.model.body("ball").pos.copy()          # 这个才是定义的home pos
        # print("init pos: ", self.init_ball_pos)
        # self.left_finger_pos = self.data.body('left_finger').xpos
        # self.right_finger_pos = self.data.body('right_finger').xpos
        self.left_finger_pos = self.data.site_xpos[self.left_site_id]
        self.right_finger_pos = self.data.site_xpos[self.right_site_id]
        self.ball_pos = self.data.xpos[self.model.body("ball").id]
        self.hand_pos = self.data.xpos[self.model.body("hand").id]
        self.at_waypoint = False

    def _get_obs(self):
        tcp_quat = self.data.xquat[self.model.body("hand").id] # 手部姿态（四元数）
        #target_quat = self.data.xquat[self.model.body("ball").id] # 目标姿态
        target_quat = np.array([0,1,0,0])  # 目标姿态固定为某个值，简化问题
        lift_height_obs = self.ball_pos[2] - self.init_ball_pos[2]
        
        return np.concatenate([
            self.data.qpos[:9].flat,  # 9个关节的位置
            self.data.qvel[:9].flat,  # 9个关节的速度
            self.hand_pos - self.ball_pos,     # 手部到目标的相对位移
            self.ball_pos,
            self.left_finger_pos,
            self.right_finger_pos,
            [lift_height_obs],
            [self.at_waypoint]
            # (R.from_quat(target_quat) * R.from_quat(tcp_quat).inv()).as_rotvec()  # 朝向差异，3维
        ]).astype(np.float64)         # 总计32维
    
    def check_robot_table_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 == self.table_id and g2 in self.robot_geom_ids) or (g2 == self.table_id and g1 in self.robot_geom_ids):
                return True
        return False
    
    def check_robot_ball_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            if(g1 == self.hand_id and g2 == self.ball_geom_id) or \
                (g2 == self.hand_id and g1 == self.ball_geom_id):
                return True
        return False
    
    def ball_is_in_gripper(self):
        gripper_midpoint = (self.left_finger_pos + self.right_finger_pos) / 2.0
        finger_to_finger = self.right_finger_pos - self.left_finger_pos
        ball_to_left = self.ball_pos - self.left_finger_pos

        mag_sq = np.sum(np.square(finger_to_finger)) + 1e-6
        s = np.dot(ball_to_left, finger_to_finger) / mag_sq

        is_between_fingers = (0.0 < s < 1.0) # s = 0.5 表示球在手指中间，s < 0或s > 1表示球在手指连线外
        is_above_ball = (self.hand_pos[2] > self.ball_pos[2] + 0.01)  # 高度检查，防止手掌低于小球靠平推小球把小球垫上去

        dist_to_midpoint = np.linalg.norm(gripper_midpoint - self.ball_pos)
        is_close_to_center = (dist_to_midpoint < 0.05)

        return bool(is_between_fingers and is_above_ball and is_close_to_center)

    def step(self, action):
        current_qpos = self.data.qpos[:7].copy()
        # print(f"DEBUG - Action min: {action.min():.4f}, max: {action.max():.4f}")
        # print(f"Action Space Low: {self.action_space.low}")
        # print(f"Action Space High: {self.action_space.high}")

        delta_qpos = action[:7] * 0.02  
        target_qpos = current_qpos + delta_qpos  # 缩小步长 使整体运动更平滑
        target_qpos = np.clip(target_qpos, self.joint_limits_low[:7], self.joint_limits_high[:7]) # 输入新的角度并相对于limit截断

        gripper_ctrl = (action[7] + 1.0) / 2.0 * 255.0 # 夹爪控制

        full_ctrl = np.concatenate([target_qpos, [gripper_ctrl]])
        self.do_simulation(full_ctrl, self.frame_skip)  # 执行动作并推进仿真
        obs = self._get_obs()    # 获取新的状态观察

        # 位置奖励：夹爪中心靠近小球中心；惩罚夹爪间距小于小球直径（由于夹取奖励，机械臂总是倾向于开局即闭合夹爪）
        # 目前将位置奖励区分为两段 悬停点和球心 当前30w步的结果是夹爪可以不碰球的情况下移到球心 下一步测试随着步数增加能否学会夹取 查看一下tensorboard有无收敛 
        # reach reward或许仍需修改 
        ball_radius = self.model.geom_size[self.ball_geom_id][0]
        gripper_midpoint = (self.left_finger_pos + self.right_finger_pos) / 2.0 
        dist = np.linalg.norm(gripper_midpoint - self.ball_pos)  
        # dist_threshold = 0.02
        # reward_reach = -2.0 * max(dist - dist_threshold, 0) # 留出学习空间 在夹爪中心点和球心非常近的情况下 自己探索一个合适的夹取位置

        waypoint_pos = self.ball_pos.copy()
        waypoint_pos[2] += 0.12
        if not self.at_waypoint:
            dist_to_waypoint = np.linalg.norm(gripper_midpoint - waypoint_pos)
            if dist_to_waypoint < 0.02:
                self.at_waypoint = True
            reward_reach = -5.0 * dist_to_waypoint
        else:
            reward_reach = -0.1 + 5.0 * (0.12 - dist)

        current_gap = np.linalg.norm(self.left_finger_pos - self.right_finger_pos)
        reward_open_gripper = 0.0
        if current_gap < 1.8 * ball_radius:
            reward_open_gripper = -0.5
            margin = 0.005
            reward_open_gripper += -10.0 * (2 * ball_radius - current_gap + margin)

        # 夹取奖励： 奖励两个夹爪与小球距离相等；奖励收紧夹爪；奖励球心与两根手指的连线平行（共线）
        reward_grasp = 0.0
        vector_l = self.left_finger_pos - self.ball_pos
        vector_r = self.right_finger_pos - self.ball_pos
        dist_l = np.linalg.norm(vector_l)
        dist_r = np.linalg.norm(vector_r)
        symmetry_error = abs(dist_l - dist_r)

        reward_parallel = 0.0
        reward_symmetry = 0.0

        if dist < 0.08:
            reward_symmetry = -10.0 * symmetry_error  # 近距离时给不对称惩罚
            if self.ball_is_in_gripper():
                v_l_unit = vector_l / (dist_l + 1e-6)
                v_r_unit = vector_r / (dist_r + 1e-6)
                dot_product = np.dot(v_l_unit, v_r_unit)
                alignment_score = (1.0 - dot_product) / 2.0
                reward_parallel = 2.0 * alignment_score        # 包络以后给共线奖励
                # reward_symmetry = alignment_score * (-2.0 * symmetry_error)

        if dist_l < 0.05 and dist_r < 0.05: 
            finger_dist = np.linalg.norm(self.left_finger_pos - self.right_finger_pos)
            reward_grasp = 10.0 * (0.08 - finger_dist)

        # 提升奖励
        # table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        # ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        # half_height = self.model.geom_size[table_id][2]
        # center_z = self.data.geom_xpos[table_id][2]
        half_height = self.model.geom_size[self.table_id][2]
        center_z = self.data.geom_xpos[self.table_id][2]
        table_top_z = center_z + half_height
        ball_height = self.ball_pos[2]
        lift_height = ball_height - (table_top_z + ball_radius)

        reward_lift = 0.0
        if self.ball_is_in_gripper():
            self.model.geom_rgba[self.ball_geom_id] = [0, 1, 0, 1]
            reward_lift = 200.0 * max(lift_height, 0.0)
            if lift_height > 0.01:
                reward_lift += 1.5
                if lift_height > 0.05:
                    reward_lift += 1.0
                    if lift_height > 0.1:
                        reward_lift += 1.5
        else:
            self.model.geom_rgba[self.ball_geom_id] = [1, 0, 0, 1]

        # 平滑奖励
        reward_ctrl = -0.1 * np.square(action[:8]).sum()

        # 撞桌惩罚及手掌触球惩罚
        collision_penalty = 0
        collision_fatal = False

        if self.check_robot_table_collision() or self.check_robot_ball_collision():
            collision_penalty = -1.0
            # collision_fatal = True

        # 惩罚z不变时的xy值变动
        horizontal_displacement = np.linalg.norm(self.ball_pos[:2] - self.init_ball_pos[:2])
        reward_anti_push = 0.0

        displacement_fatal = False
        if lift_height < 0.01:
            # if horizontal_displacement > 0.05:
                # reward_anti_push = -10.0 * horizontal_displacement
                ball_qvel_adr = self.model.jnt_dofadr[self.ball_joint_id]
                ball_vel_3d = self.data.qvel[ball_qvel_adr : ball_qvel_adr + 3]
                ball_vel = np.linalg.norm(ball_vel_3d)
                reward_anti_push = -1.0 * ball_vel
                reward_anti_push += -5.0 * horizontal_displacement    # TODO: try to solve displacement problem, robot arm shouldn't "attack" the ball to make it in the appropriate position
                if horizontal_displacement > 0.5:
                    displacement_fatal = True
                    reward_anti_push = -50.0

        reward = reward_reach + reward_open_gripper + reward_symmetry + reward_parallel + reward_grasp + reward_lift + collision_penalty + reward_ctrl + reward_anti_push # 计算奖励：这里使用距离的相反数，距离越近，奖励越大；同时对动作的大小进行惩罚，鼓励更小的动作

        terminated = bool(lift_height > 0.1) or collision_fatal or displacement_fatal
        truncated = False

        #info = {"distance": general_dist}
        info = {
            "is_success": bool(lift_height > 0.1),
            "lift_height": lift_height,
            "dist": dist,
            "charts/reward_reach": reward_reach,
            "charts/reward_open_gripper": reward_open_gripper,
            "charts/reward_symmetry": reward_symmetry,
            "charts/reward_parallel": reward_parallel,
            "charts/reward_grasp": reward_grasp,
            "charts/reward_lift": reward_lift,
            "charts/reward_anti_push": reward_anti_push,
            "charts/reward_ctrl": reward_ctrl,
            "charts/reward_collison": collision_penalty
        }

        if self.render_mode == "human":   # 渲染图像
             self.render()

        return obs, reward, terminated, truncated, info
    
    # 重置逻辑，每当机器人完成任务或失败后，调用此函数把世界“复位”
    def reset_model(self):
        qpos = self.model.key_qpos[0].copy()    # home位置
        qvel = self.model.key_qvel[0].copy()

        ball_joint_adr = self.model.joint("ball_joint").qposadr[0]
        qpos[ball_joint_adr : ball_joint_adr +3] = [0.5, 0.0, 0.45]

        #self.data.xpos[self.model.body("ball").id] = np.array([0.5, 0, 0.45]) 
        
        self.set_state(qpos, qvel)
        self.at_waypoint = False
        return self._get_obs()
    
# env = PandaReachEnv(render_mode = "human")
# check_env(env)
# ctrl, open_gripper, reach