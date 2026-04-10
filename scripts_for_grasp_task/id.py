import mujoco
import numpy as np


ARM_DOF = 7

class IDSolver:
    def __init__(self):
        pass

    @staticmethod
    def solve_id_autonomous(robot, target_q, target_dq, target_ddq):
        q_backup = np.copy(robot.data.qpos)
        dq_backup = np.copy(robot.data.qvel)

        robot.data.qpos[:ARM_DOF] = target_q
        robot.data.qvel[:ARM_DOF] = target_dq
        robot.data.qacc[:ARM_DOF] = target_ddq
        
        mujoco.mj_inverse(robot.model, robot.data)

        tau_id = np.copy(robot.data.qfrc_inverse[:ARM_DOF])

        robot.data.qpos[:] = q_backup
        robot.data.qvel[:] = dq_backup

        return tau_id
    
    def solve_id_manual(self, robot, target_q, target_dq, target_ddq):
        total_tau = np.zeros(ARM_DOF)
        g = np.array([0, 0, -9.81])

        for i in range(1, 1+ARM_DOF):
            body_id = i
            mass = robot.model.body_mass[body_id]

            a_COM = robot.data.cacc[body_id][3:6]
            F_i = mass * (a_COM - g)

            jacp = np.zeros((3, robot.model.nv))
            jacr = np.zeros((3, robot.model.nv))
            mujoco.mj_jac(robot.model, robot.data, jacp, jacr, robot.data.xipos[body_id], body_id)
            J_transition = jacp[:, :7]

            omega = robot.data.cvel[body_id][:3] #角速度
            alpha = robot.data.cacc[body_id][:3] #角加速度
            I_i = np.diag(robot.model.body_inertia[body_id])  #局部坐标系下的惯量矩
            R = robot.data.ximat[body_id].reshape(3,3)
            I_i = R @ I_i @ R.T  # 局部坐标系切换到世界坐标系

            M_i = I_i @ alpha + np.cross(omega, I_i @ omega)
            J_rotation = jacr[:, :7]

            total_tau += J_transition.T @ F_i + J_rotation.T @ M_i


    def custom_solve_id_half_autonomous(model, data, target_q, target_dq, target_ddq):
        """
        基于雅可比投影法的自定义逆动力学求解器 (Newton-Euler)
        """
        # 1. 状态备份
        q_backup = np.copy(data.qpos)
        dq_backup = np.copy(data.qvel)
        acc_backup = np.copy(data.qacc)

        # 2. 全局清零与状态注入 (防止夹爪等末端状态污染)
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qacc[:] = 0

        data.qpos[:ARM_DOF] = target_q
        data.qvel[:ARM_DOF] = target_dq
        data.qacc[:ARM_DOF] = target_ddq

        # 3. 仅更新运动学状态 (不调用 mj_inverse，只更新位置、速度、加速度相关的雅可比和空间变量)
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        mujoco.mj_fwdAcceleration(model, data) 

        # 初始化我们手算的关节力矩数组
        tau_manual = np.zeros(model.nv)
        
        # 获取系统重力 (通常是 [0, 0, -9.81])
        g_0 = model.opt.gravity 

        # 4. 遍历所有连杆 (跳过 worldbody, id=0)
        for i in range(1, model.nbody):
            # 如果连杆没有质量，跳过计算
            mass = model.body_mass[i]
            if mass < 1e-6:
                continue

            # --- A. 获取 MuJoCo 为我们算好的运动学数据 ---
            # data.cacc: CoM 的空间加速度 [角加速度(3), 线加速度(3)] (世界坐标系)
            alpha_0 = data.cacc[i, 0:3]  
            a_CoM_0 = data.cacc[i, 3:6]  

            # data.cvel: CoM 的空间速度 [角速度(3), 线速度(3)] (世界坐标系)
            omega_0 = data.cvel[i, 0:3]
            
            # 获取惯性张量 (世界坐标系)
            # model.body_inertia 是主惯量 (对角阵)
            # data.ximat 是主惯量坐标系到世界坐标系的旋转矩阵 R
            R_i0 = data.ximat[i].reshape(3, 3)
            I_local = np.diag(model.body_inertia[i])
            I_0 = R_i0 @ I_local @ R_i0.T

            # 获取当前连杆质心 (CoM) 对应的雅可比矩阵 (世界坐标系)
            jacp = np.zeros((3, model.nv)) # 线速度雅可比 J_v
            jacr = np.zeros((3, model.nv)) # 角速度雅可比 J_w
            mujoco.mj_jacBodyCom(model, data, jacp, jacr, i)

            # --- B. Newton-Euler 力学计算 (对应你提供的公式) ---
            
            # 1. 牛顿平移部分: F = m * (a - g)
            # 注意: 你的代码里是在力里面减去重力，等效于给了连杆一个向上的反作用力
            F_0 = mass * (a_CoM_0 - g_0)

            # 2. 欧拉旋转部分: Tau = I * alpha + omega x (I * omega)
            # 你的代码里有 Li_dot = Theta @ alpha + cross(...)，这里完全对应
            # 另外，由于我们在质心(CoM)处计算，你的代码里的 mra (偏心加速度矩) 和 M_gravity_i 在质心处为 0，所以可以省去！
            Tau_0 = I_0 @ alpha_0 + np.cross(omega_0, I_0 @ omega_0)

            # --- C. 雅可比投影 (添加到总力矩) ---
            # Q_t + Q_r = J_v^T @ F + J_w^T @ Tau
            tau_manual += jacp.T @ F_0 + jacr.T @ Tau_0

        # 5. 状态还原
        data.qpos[:] = q_backup
        data.qvel[:] = dq_backup
        data.qacc[:] = acc_backup
        mujoco.mj_forward(model, data) # 还原主循环状态

        # 返回前 ARM_DOF 个关节的手算力矩
        return tau_manual[:ARM_DOF]