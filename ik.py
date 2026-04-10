import numpy as np
import mujoco
from enum import Enum
import robot 
from scipy.spatial.transform import Rotation as R

ARM_DOF = 7
 
IKOptions = Enum('IKOptions', ['COMFORT', 'LIMIT', 'BOTH'])

class IKSolverVelocity:
    def __init__(self, W: np.array, k_position: float, \
                 k_orientation: float, robot, alpha: float = 0.5, eta: float = 1.0, opt=IKOptions.LIMIT) -> None:
        self.W = W
        self.W_inv = np.linalg.inv(W)
        self.k_position = k_position
        self.dt=robot.model.opt.timestep
        self.k_orientation = k_orientation
        self.alpha = alpha     #调节躲避动作有多“积极”的增益
        self.eta = eta         #调节避障势能场强度的增益
        self.opt = opt


        try:
            self.hand_id = robot.model.body('hand').id
            self.ball_id = robot.model.body('ball').id
        except ValueError as e:
            print(f"Error: {e}")
            print([robot.model.body(i).name for i in range(robot.model.nbody)])
            exit()

    # 计算关节限位
    def H_limit(self):
        q = robot.data.qpos[:ARM_DOF]
        H = 0.0
        for i in range(len(q)):
            q_min, q_max = robot.model.jnt_range[i]
            range_sq = (q_max - q_min)**2
            H += ((q[i] - q_min)**2 + (q[i] - q_max)**2)/ range_sq
        return H
    
    def grad_H_limit(self, robot):
        q = robot.data.qpos[:ARM_DOF]
        grad_H = np.zeros(ARM_DOF)
        for i in range(len(q)):
            q_min, q_max = robot.model.jnt_range[i]

            ###临时补丁，写死第二个关节的限位，防止机械臂撞桌
            if i ==1:
                q_min = -0.1
                q_max = 0.1
            ###

            range_sq = (q_max - q_min)**2
            grad_H[i] = 2 * (q[i] - q_min + q[i] - q_max) / range_sq
        return grad_H

    def H_comfort(self, robot):
        q = robot.data.qpos[:ARM_DOF]
        H = np.sum(0.5 * (q - robot.q_comfort)**2)
        return H

    def grad_H_comfort(self, robot):
        q = robot.data.qpos[:ARM_DOF]
        grad_H = q - robot.q_comfort
        return grad_H

    #获取所有中间links的名字，用于计算H_obstacle, 这里去除link0(基座)，link1(离基座很近，基本无需做避障处理)，link7（末端link，不避障）
    def get_middle_link_names(self, robot):
        all_links = sorted([
            mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, i) 
            for i in range(robot.model.nbody) 
                if "link" in (mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
        ])
        
        obs_links = all_links[2:-1]
        #print(f"当前监控的避障连杆: {obs_links}")
        return obs_links

    def H_obstacle(self, robot):
        H = 0.0
        obs_links = self.get_middle_link_names(robot)
        for name in obs_links:
            body_id = self.model.body(name).id
            pos = self.data.xpos[body_id]
            
            obstacle_pos = robot.data.xpos[self.ball_id]
            diff = pos - obstacle_pos
            dist = np.linalg.norm(diff)
            
            avoidance_radius = 0.15
            if dist < avoidance_radius and dist > 0.001:
                H += 0.5 * self.eta * (1.0/dist - 1.0/avoidance_radius)**2
        return H        

    # 计算避障限位
    def grad_H_obstacle(self, robot):
        grad_H = np.zeros(ARM_DOF)
        
        obs_links = self.get_middle_link_names(robot)
        for name in obs_links:
            body_id = robot.model.body(name).id
            body_pos = robot.data.xpos[body_id]
            
            obstacle_pos = robot.data.xpos[self.ball_id]
            diff = body_pos - obstacle_pos
            dist = np.linalg.norm(diff)
            
            avoidance_radius = 0.15
            if dist < avoidance_radius and dist > 0.001:
                # 空间排斥力 (Artificial Potential Field)
                unit_vec = diff / dist            #指向障碍物的单位向量
                mag = self.eta * (1.0/dist - 1.0/avoidance_radius) * (1.0/dist**2)  #力的大小
                f_rep = mag * unit_vec  
                
                # 获取该连杆的雅可比
                jac_p = np.zeros((3, robot.model.nv))
                mujoco.mj_jac(robot.model, robot.data, jac_p, None, body_pos, body_id)
                
                # 累加到关节梯度：J.T @ F
                grad_H += jac_p[:, :7].T @ f_rep
        return grad_H



    def solve_ik_only_transition(self, robot, target_pos):
        #获取当前状态
        current_pos = robot.data.xpos[self.hand_id]
        current_orientation = robot.data.xmat[self.hand_id].reshape(3, 3)


        #设定目标状态
        error_pos = target_pos - current_pos
        #target_orientation = current_orientation

        jacp = np.zeros((3, robot.model.nv))
        mujoco.mj_jac(robot.model, robot.data, jacp, None, current_pos, self.hand_id)

        k_position = 2.0
        #k_orientation = 1.0

        #J_full = np.vstack((jacp, jacr)) 
        J = jacp[:, :7]  # 取前7列对应机械臂关节
        J_pseudo = self.W_inv @ J.T @ np.linalg.inv(J @ self.W_inv @ J.T + 1e-4 * np.eye(3)) # 加小量防止奇异
        n = ARM_DOF
        N_W = np.eye(n) - J_pseudo @ J

        body_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            robot.model, 
            robot.data,
            mujoco.mjtObj.mjOBJ_BODY,  #类别为BODY 
            self.hand_id,                   #指定一个具体BODY
            body_vel,                  #存结果
            0                          #世界坐标系
            ) #获取手部线速度和角速度
    
        v_eff = k_position * error_pos  

        grad_H_obstacle = self.grad_H_obstacle(robot)
        #grad_H_limit = self.grad_H_limit(robot)
        grad_H_comfort = self.grad_H_comfort(robot)
        grad_H_total = grad_H_obstacle + grad_H_comfort
        dq = J_pseudo @ v_eff - self.alpha * N_W @ self.W_inv @ grad_H_total.T
        return dq
    
    def solve_ik(self, robot, target_pos, target_orientation):
        #获取当前状态
        current_pos = robot.data.xpos[self.hand_id]
        current_orientation = robot.data.xmat[self.hand_id].reshape(3, 3)
        #current_orientation = robot.data.xquat[self.hand_id]

        error_pos = target_pos - current_pos
        #target_orientation = np.diag((1,-1,-1))

        jacp = np.zeros((3, robot.model.nv))
        jacr = np.zeros((3, robot.model.nv))
        mujoco.mj_jac(robot.model, robot.data, jacp, jacr, current_pos, self.hand_id)

        J_full = np.vstack((jacp, jacr)) 
        J = J_full[:, :7]  # 取前7列对应机械臂关节
        J_pseudo = self.W_inv @ J.T @ np.linalg.inv(J @ self.W_inv @ J.T + 1e-4 * np.eye(6)) # 加小量防止奇异
        n = ARM_DOF
        N_W = np.eye(n) - J_pseudo @ J

        body_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            robot.model, 
            robot.data,
            mujoco.mjtObj.mjOBJ_BODY,  #类别为BODY 
            self.hand_id,                   #指定一个具体BODY
            body_vel,                  #存结果
            0                          #世界坐标系
            ) #获取手部线速度和角速度
    
        #v_eff = body_vel[3:6] + k_position * (target_pos - current_pos)  #线速度 = 当前线速度 + 位置误差项
        v_eff = self.k_position * error_pos  
        #R_err = current_orientation @ target_orientation.T
        R_err = target_orientation @ current_orientation.T
        #w_eff = body_vel[0:3] + k_orientation * (current_orientation.T @ (R.from_matrix(R_err).as_rotvec()))  #角速度 = 当前角速度 + 姿态误差项
        w_eff = self.k_orientation * (R.from_matrix(R_err).as_rotvec())
        workapace_vel_eff = np.hstack((v_eff, w_eff))

        grad_H_obstacle = self.grad_H_obstacle(robot)
        grad_H_limit = self.grad_H_limit(robot)
        grad_H_comfort = self.grad_H_comfort(robot)
        grad_H_total = grad_H_obstacle
        dq = J_pseudo @ workapace_vel_eff + self.alpha * N_W @ self.W_inv @ grad_H_total.T
        return dq

    def solve_ik_with_reduced_ori_constraint(self, robot, target_pos, target_z_axis=np.array([0,0,-1])):
        current_pos = robot.data.xpos[self.hand_id]
        current_orientation = robot.data.xmat[self.hand_id].reshape(3,3)

        #获取夹爪当前Z轴方向
        current_z_axis = current_orientation[:,2]

        error_pos = target_pos - current_pos

        jacp = np.zeros((3, robot.model.nv))
        jacr = np.zeros((3, robot.model.nv))
        mujoco.mj_jac(robot.model, robot.data, jacp, jacr, current_pos, self.hand_id)

        J_full = np.vstack((jacp, jacr))
        J = J_full[:, :7]
        J_5 = np.delete(J, 5, axis=0)  # 删除第6行（索引5），对应Z轴旋转约束
        J_pseudo = self.W_inv @ J_5.T @ np.linalg.inv(J_5 @ self.W_inv @ J_5.T + 1e-4 * np.eye(5)) # 加小量防止奇异
        n = ARM_DOF
        N_W = np.eye(n) - J_pseudo @ J_5

        v_eff = self.k_position * error_pos
        w_eff = self.k_orientation * np.cross(current_z_axis, target_z_axis)  # 只约束Z轴方向，使用叉积计算旋转误差
        workapace_vel_eff = np.delete(np.hstack((v_eff, w_eff)), 5)  # 删除对应Z轴旋转的速度分量

        grad_H_total = self.grad_H_obstacle(robot) + self.grad_H_limit(robot)
        dq = J_pseudo @ workapace_vel_eff + self.alpha * N_W @ self.W_inv @ grad_H_total.T
        return dq

    # For test 
    def solve_ik_only_orientation(self, robot, target_orientation):
        #获取当前状态
        current_pos = robot.data.xpos[self.hand_id]
        current_orientation = robot.data.xmat[self.hand_id].reshape(3, 3)
        #current_orientation = robot.data.xquat[self.hand_id]

        jacp = np.zeros((3, robot.model.nv))
        jacr = np.zeros((3, robot.model.nv))
        mujoco.mj_jac(robot.model, robot.data, jacp, jacr, current_pos, self.hand_id)

        J_full = np.vstack((jacp, jacr)) 
        J = jacr[:, :7]  # 取前7列对应机械臂关节
        J_pseudo = self.W_inv @ J.T @ np.linalg.inv(J @ self.W_inv @ J.T + 1e-4 * np.eye(3)) # 加小量防止奇异
        n = ARM_DOF
        N_W = np.eye(n) - J_pseudo @ J

        #v_eff = body_vel[3:6] + self.k_position * (target_pos - current_pos)  #线速度 = 当前线速度 + 位置误差项
        #v_eff = self.k_position * error_pos  
        #R_err = current_orientation @ target_orientation.T
        R_err = target_orientation @ current_orientation.T
        #w_eff = body_vel[0:3] + self.k_orientation * (current_orientation.T @ (R.from_matrix(R_err).as_rotvec()))  #角速度 = 当前角速度 + 姿态误差项
        w_eff = self.k_orientation * (R.from_matrix(R_err).as_rotvec())
        #workapace_vel_eff = np.hstack((v_eff, w_eff))

        grad_H_obstacle = self.grad_H_obstacle(robot)
        grad_H_limit = self.grad_H_limit(robot)
        grad_H_total = 2 * grad_H_obstacle + grad_H_limit
        dq = J_pseudo @ w_eff + self.alpha * N_W @ self.W_inv @ grad_H_total.T
        dq = J_pseudo @ w_eff
        return dq
    
class IKSolverPosition:
    def __init__(self, robot, alpha: float=0.5 , position_factor: float=1.0, weights: np.array=np.diag([2,5,2,1,1,1,1]), rotation_factor: float=1.0,  maxiter: int = 100, tol: float = 0.01):
        self.maxiter = maxiter
        self.maxerror = tol
        self.position_factor = position_factor
        self.rotation_factor = rotation_factor
        self.W_inv = np.linalg.inv(weights)
        self.alpha = alpha

        try:
            self.hand_id = robot.model.body('hand').id
            self.ball_id = robot.model.body('ball').id
        except ValueError as e:
            print(f"Error: {e}")
            print([robot.model.body(i).name for i in range(robot.model.nbody)])
            exit()

    def get_middle_link_names(self, robot):
        all_links = sorted([
            mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, i) 
            for i in range(robot.model.nbody) 
                if "link" in (mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
        ])
        
        obs_links = all_links[2:-1]
        #print(f"当前监控的避障连杆: {obs_links}")
        return obs_links

    def grad_H_obstacle(self, robot):
        grad_H = np.zeros(ARM_DOF)
        
        obs_links = self.get_middle_link_names(robot)
        for name in obs_links:
            body_id = robot.model.body(name).id
            body_pos = robot.data.xpos[body_id]
            
            obstacle_pos = robot.data.xpos[self.ball_id]
            diff = body_pos - obstacle_pos
            dist = np.linalg.norm(diff)
            
            avoidance_radius = 0.15
            if dist < avoidance_radius and dist > 0.001:
                # 空间排斥力 (Artificial Potential Field)
                unit_vec = diff / dist            #指向障碍物的单位向量
                mag = self.eta * (1.0/dist - 1.0/avoidance_radius) * (1.0/dist**2)  #力的大小
                f_rep = mag * unit_vec  
                
                # 获取该连杆的雅可比
                jac_p = np.zeros((3, robot.model.nv))
                mujoco.mj_jac(robot.model, robot.data, jac_p, None, body_pos, body_id)
                
                # 累加到关节梯度：J.T @ F
                grad_H += jac_p[:, :7].T @ f_rep
        return grad_H
    
    def grad_H_comfort(self, robot):
        q = robot.data.qpos[:ARM_DOF]
        grad_H = q - robot.q_comfort
        return grad_H
        
    def calculate_joint_angles(self, robot, q_start, target_pos, target_orientation, dt):
        q = np.copy(q_start)
        counter = 0

        qpos_backup = np.copy(robot.data.qpos)

        while True:
            robot.data.qpos[:ARM_DOF] = q
            mujoco.mj_forward(robot.model, robot.data)

            jacp = np.zeros((3, robot.model.nv))
            jacr = np.zeros((3, robot.model.nv))
            current_pos = robot.data.xpos[self.hand_id]
            current_orientation = robot.data.xmat[self.hand_id].reshape(3, 3)
            mujoco.mj_jac(robot.model, robot.data, jacp, jacr, current_pos, self.hand_id)

            J_full = np.vstack((jacp, jacr)) 
            J = J_full[:, :7]  # 取前7列对应机械臂关节
            J_pseudo = J.T @ np.linalg.inv(J @ J.T + 1e-2 * np.eye(6)) # 加小量防止奇异
        
            error_pos = target_pos - current_pos
            v_eff = self.position_factor * error_pos  
            #R_err = current_orientation @ target_orientation.T
            R_err = target_orientation @ current_orientation.T
            rot_err = R.from_matrix(R_err).as_rotvec()
            w_eff = self.rotation_factor * (rot_err)
            workspace_vel_eff = np.hstack((v_eff, w_eff))

            q += J_pseudo @ workspace_vel_eff

            counter += 1

            if(counter > self.maxiter):
                print("\nwarning: IKSolverPositionDLS reached maximum number of iterations!\n")
                break

            if self.position_factor * np.linalg.norm(error_pos) \
               +self.rotation_factor * np.linalg.norm(rot_err) < self.maxerror :
                break 
        robot.data.qpos[:] = qpos_backup
        mujoco.mj_forward(robot.model, robot.data)

        return q
    
    def calculate_joint_angles_only_transition(self, robot, q_start, target_pos):
        q = np.copy(q_start)
        counter = 0

        qpos_backup = np.copy(robot.data.qpos)

        while True:
            robot.data.qpos[:ARM_DOF] = q
            mujoco.mj_forward(robot.model, robot.data)

            current_pos = robot.data.xpos[self.hand_id]
            error_pos = target_pos - current_pos

            jacp = np.zeros((3, robot.model.nv))
            mujoco.mj_jac(robot.model, robot.data, jacp, None, current_pos, self.hand_id)

            J = jacp[:, :7]  # 取前7列对应机械臂关节
            J_pseudo = self.W_inv @ J.T @ np.linalg.inv(J @ self.W_inv @ J.T + 1e-2 * np.eye(3)) # 加小量防止奇异
            n = ARM_DOF
            N_W = np.eye(n) - J_pseudo @ J

            body_vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                robot.model, 
                robot.data,
                mujoco.mjtObj.mjOBJ_BODY,  #类别为BODY 
                self.hand_id,                   #指定一个具体BODY
                body_vel,                  #存结果
                0                          #世界坐标系
                ) #获取手部线速度和角速度
        
            v_eff = self.position_factor * error_pos  

            grad_H_obstacle = self.grad_H_obstacle(robot)
            #grad_H_limit = self.grad_H_limit(robot)
            grad_H_comfort = self.grad_H_comfort(robot)
            grad_H_total = grad_H_obstacle + grad_H_comfort
            dq = J_pseudo @ v_eff - self.alpha * N_W @ self.W_inv @ grad_H_total.T

            q += dq

            counter += 1

            if(counter > self.maxiter):
                print("\nwarning: IKSolverPositionDLS reached maximum number of iterations!\n")
                break

            if self.position_factor * np.linalg.norm(error_pos) < self.maxerror :
                break 
        robot.data.qpos[:] = qpos_backup
        mujoco.mj_forward(robot.model, robot.data)
        return q