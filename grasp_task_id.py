import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from enum import Enum

import ik
import id
import planning.PTP as ptp
import robot


### Setting start
ARM_DOF = 7
PlanningOptions = Enum('PlanningOptions', ['PTPJOINTSPACE', 'PTPWORKSPACE'])

TIME_INTEGRATION = Enum('SOLVER_OPTIONS', ['euler', 'rk45'])
SCENARIO = Enum('SCENARIO', ['integration' , 'real'])

substeps = 1 # steps per inported time step
scenario = SCENARIO.integration

if scenario == SCENARIO.integration:
    time_integration = TIME_INTEGRATION.rk45
    pd_controller = False
    # p and d gain
    #kp = 1000 * np.eye(ARM_DOF)
    kp = np.diag([2000, 2000, 2000, 2000, 500, 500, 200])
    #kd = 0.1 * np.eye(ARM_DOF)
    kd = np.diag([80, 80, 80, 60, 20, 15, 5])
    # disturbance False/True and standard deviation
    disturbance = False
    std = 0.1
    disturbance_rate = 10 # noise all x timesteps
elif scenario == SCENARIO.real:
    time_integration = TIME_INTEGRATION.euler
    pd_controller = True
    kp = 10 * np.eye(ARM_DOF)
    kd = 0.1 * np.eye(ARM_DOF)
    disturbance = True
    std = 0.1
    disturbance_rate = 10
else:
    raise NotImplementedError('scenario not implemented')
### Setting end


current_dir = os.path.dirname(os.path.abspath(__file__))
#xml_path = os.path.join(current_dir, 'grasp_task_without_table.xml')
xml_path = os.path.join(current_dir, 'grasp_task_id.xml')

#加载模型
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
rob = robot.Robot(model, data)
data.qpos[:9] = model.key_qpos[0,:9]
#data.ctrl[:8] = model.key_ctrl[0, :8]
data.qvel[:9] = 0
mujoco.mj_forward(model, data)

weights = np.diag([2,5,2,1,1,1,1])   #数值越大 越不倾向于动
k_position = 3.0
k_orientation = 3.0

solver = ik.IKSolverVelocity(weights, k_position, k_orientation, rob)
dt = solver.dt
#solver.get_middle_link_names(rob)

#启动可视化器
with mujoco.viewer.launch_passive(model, data) as viewer:
    phase = 0
    mujoco.mj_forward(model, data)   #刷新一次 确保init_xpos的正确性

    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grab_weld")

    ball_pos = rob.data.xpos[solver.ball_id]
    hand_pos = rob.data.xpos[solver.hand_id]

    init_xpos = hand_pos #初始位置/home位置
    hover_target = ball_pos + np.array([-0.02, 0.0, 0.2])  # 一阶段悬停位置
    #grab_target = ball_pos + np.array([0.0, 0.0, 0.08])  # 二阶段抓球位置
    grab_target = ball_pos + np.array([0.0002, 0., 0.07])
    lift_target = ball_pos + np.array([0, 0, 0.2])  # 三阶段抬起位置

    through_points = np.vstack([init_xpos, hover_target, grab_target, lift_target])  #轨迹规划路径点 该点位无变化 选择在循环外赋值

    num_waypoints = len(through_points)
    waypoints_jointspace = np.zeros((num_waypoints, ARM_DOF))
    q_start = rob.data.qpos[:ARM_DOF].copy()

    ik_solver_position = ik.IKSolverPosition(rob)
    current_q_for_ik_position = q_start.copy()
    target_orientation = np.diag((1.0, -1.0, -1.0))
    
    for i in range(num_waypoints):
        if i == 0:
            res_q = np.array((0.,0.,0.,-1.57079, 0, 1.57079, -0.7853))
        elif i == 1:
            res_q = ik_solver_position.calculate_joint_angles_only_transition(rob, current_q_for_ik_position, through_points[i,:])
        else:
            res_q = ik_solver_position.calculate_joint_angles(rob, current_q_for_ik_position, through_points[i,:], target_orientation, dt)
        waypoints_jointspace[i,:] = res_q
        current_q_for_ik_position = res_q.copy()

    all_q = []
    all_dq = []
    all_ddq = []

    phases = [
        # 一阶段 从home点到上方悬停点
        {'start_idx': 0, 'end_idx': 1, 'method': 'PTPJOINTSPACE', 'speed': 'fast'},

        # 二阶段 打开夹爪
        {'start_idx': 1, 'end_idx': 1, 'method': 'PTPJOINTSPACE', 'speed': 'fast'},

        # 三阶段 从悬停点到抓球点 
        {'start_idx': 1, 'end_idx': 2, 'method': 'PTPJOINTSPACE', 'speed': 'fast'},

        # 四阶段 收紧夹爪 位置不变
        {'start_idx': 2, 'end_idx': 2, 'method': 'PTPWORKSPACE', 'speed': 'slow'},

        # 五阶段 夹爪上移 停住
        {'start_idx': 2, 'end_idx': 3, 'method': 'PTPWORKSPACE', 'speed': 'slow'}
    ]
    fast_time = 1.5
    slow_time = 2.5

    for phase in phases:
        start_idx = phase['start_idx']
        end_idx = phase['end_idx']
        method = phase['method']
        speed = phase['speed']
        
        T_val = fast_time if phase['speed'] == 'fast' else slow_time
        T_part = np.ones(max(1, end_idx - start_idx)) * T_val

        ### 对于夹取阶段特殊处理
        if start_idx == end_idx:
            wait_time = 1
            num_wait_steps = int(wait_time / dt)

            q_static = waypoints_jointspace[start_idx]

            q_seg = np.tile(q_static, (num_wait_steps, 1))
            dq_seg = np.zeros((num_wait_steps, ARM_DOF))
            ddq_seg = np.zeros((num_wait_steps, ARM_DOF))

            all_q.append(q_seg)
            all_dq.append(dq_seg)
            all_ddq.append(ddq_seg)
            continue
        ###

        # 计算所需力矩，需要有q, q', q''
        if method == 'PTPJOINTSPACE':
            phase_waypoints = waypoints_jointspace[start_idx : end_idx + 1, :]
            [T, q_seg, dq_seg, ddq_seg] = ptp.PTP_quintic(phase_waypoints, T_part, dt)
            all_q.append(q_seg.T)
            all_dq.append(dq_seg.T)
            all_ddq.append(ddq_seg.T)

        elif method == 'PTPWORKSPACE':
            phase_waypoints = through_points[start_idx: end_idx+1, :]
            [T, x_seg, dx_seg, ddx_seg] = ptp.PTP_quintic(phase_waypoints, T_part, dt)

            q_current = waypoints_jointspace[start_idx].copy()
            q_seg = []
            dq_seg = []

            for i in range(len(T)):
                target_orientation = np.diag((1, -1, -1))

                # 把当前的q喂给机器人以更新雅可比矩阵
                rob.data.qpos[:7] = q_current
                mujoco.mj_forward(rob.model, rob.data)

                dq_step = solver.solve_ik(rob, x_seg.T[i], target_orientation)
                q_current += dq_step * dt
                
                q_seg.append(q_current.copy())
                dq_seg.append(dq_step.copy())
                
            all_q.append(np.array(q_seg))
            all_dq.append(np.array(dq_seg))
            all_ddq.append(np.zeros((len(q_seg),ARM_DOF)))
        
        else:
            raise NotImplementedError('method not implemented')
        
    q_total = np.vstack(all_q)
    dq_total = np.vstack(all_dq)
    ddq_total = np.vstack(all_ddq)

    ###  先用轨迹规划得出q, q', q'' / x, x', x'' 
    ###  一般情况下，目标轨迹是用工作空间表示/x，轨迹规划可以得到x'和x''
    ###  如果某些运动片段对于工作空间里的位移过程没有要求，只对起始点位置要求，轨迹规划可以针对于关节空间，方便关节活动，此时输入的关节空间坐标由IKSolverPosition类中的calculate_joint_angles计算
    ###  逆运动学计算关节角度，关节角速度，角加速度，传递给动力学解算器解算
    ###  动力学用pd控制，kp乘q的位姿误差，kd乘q的速度误差，加上前馈的力矩：惯性力，科式力，重力

    ### 纠正被轨迹计算污染的初始位姿
    data.qpos[:ARM_DOF] = q_total[0]
    data.qvel[:ARM_DOF] = 0
    data.time = 0.0
    mujoco.mj_forward(model, data)

    flag = 0
    while viewer.is_running():
        step_start = time.time()
        #print("Initial error:", q_total[0] - data.qpos[:7])

        current_step = int(data.time / dt)

        print(f"夹爪真实出力7:{data.actuator_force[7]}")

        if current_step < len(q_total):
            q_d = q_total[current_step]
            dq_d = dq_total[current_step]
            ddq_d = ddq_total[current_step]

            tau_id = id.IDSolver.solve_id_autonomous(rob, q_d, dq_d, ddq_d)
            #tau_id = id.IDSolver.solve_id_autonomous(rob, np.array((0., 0., 0., 0., 0., 0., 0.)), 0, 0)  如果初始位姿设定为完全竖直的奇异位姿 解算出的力矩会非常大 不可用
            #tau_id = id.IDSolver.custom_solve_id_half_autonomous(rob.model, rob.data, np.array([0., 0., 0., -1.57079, 0., 1.57079, -0.7853]), 0, 0)

            error_qpos = q_d - data.qpos[:ARM_DOF]
            error_qvel = dq_d - data.qvel[:ARM_DOF]

            tau_antrieb = tau_id + kp @ error_qpos + kd @ error_qvel
            if(flag%2 == 1 and flag < 200):
                #print("tau_antrieb: ", tau_antrieb)
                pass

            data.ctrl[:7] = tau_antrieb

            if current_step > fast_time / dt and current_step < (2 * fast_time + slow_time) / dt:
                data.ctrl[7] = 255.0
            else:
                data.ctrl[7] = 0.0

        else:
            q_d = q_total[-1]
            dq_d = dq_total[-1]
            tau_id = id.IDSolver.solve_id_autonomous(rob, q_total[-1], np.zeros(ARM_DOF), np.zeros(ARM_DOF))
            error_qpos = q_d - data.qpos[:ARM_DOF]
            error_qvel = dq_d - data.qvel[:ARM_DOF]
            tau_antrib = tau_id + kp @ error_qpos + kd @ error_qvel
            data.ctrl[:7] = tau_antrib

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))