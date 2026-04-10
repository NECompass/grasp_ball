import mujoco
import mujoco.viewer
import numpy as np
import time
import os

import ik
import robot

current_dir = os.path.dirname(os.path.abspath(__file__))
#xml_path = os.path.join(current_dir, 'grasp_task_without_table.xml')
xml_path = os.path.join(current_dir, 'grasp_task_ik.xml')

#加载模型
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
rob = robot.Robot(model, data)

weights = np.diag([2,5,2,1,1,1,1])   #数值越大 越倾向于不动
k_position = 3.0
k_orientation = 3.0

solver = ik.IKSolverVelocity(weights, k_position, k_orientation, rob)
#solver.get_middle_link_names(rob)

#启动可视化器
with mujoco.viewer.launch_passive(model, data) as viewer:
    phase = 0
    while viewer.is_running():
        step_start = time.time()
        
        ball_pos = rob.data.xpos[solver.ball_id]
        hand_pos = rob.data.xpos[solver.hand_id]

        hover_target = ball_pos + np.array([0, 0, 0.3])  # 一阶段悬停位置
        grab_target = ball_pos + np.array([0.05, -0.03, 0.09])  # 二阶段抓球位置
        lift_target = ball_pos + np.array([0, 0, 0.4])  # 三阶段抬起位置
        
        if phase == 0:
            target_pos = hover_target
            target_orientation = np.diag((1,-1,-1))
            data.ctrl[7] = 0.0

            hand_quat = rob.data.xquat[solver.hand_id]
            target_quat = [0,1,0,0]
            quat_error = 1.0 - abs(np.dot(hand_quat, target_quat))
        
            if abs(hover_target[2] - hand_pos[2]) < 0.01 and \
               abs(hover_target[1] - hand_pos[1]) < 0.01 and \
               abs(hover_target[0] - hand_pos[0]) < 0.01 and \
               quat_error < 0.02 :
                phase = 1
                print("第一段完成：到达悬停点，打开夹爪，准备下探")
        
        elif phase == 1:
            
            target_pos = grab_target
            data.ctrl[7] = 255.0

            if abs(grab_target[2] - hand_pos[2]) < 0.03 and \
               abs(grab_target[1] - hand_pos[1]) < 0.03 and \
               abs(grab_target[0] - hand_pos[0]) < 0.03 :
                data.ctrl[7] = 0.0
                grab_time = data.time
                lock_pos = np.copy(hand_pos)
                phase = 2

        elif phase == 2:
            target_pos = lock_pos
            data.ctrl[7] = 0.0
            if data.time - grab_time > 1.0:  # 等待夹爪完全闭合
                phase = 3

        elif phase == 3:
            target_pos = lift_target


        if phase == 0 or phase == 1:
            dq = solver.solve_ik_only_transition(rob, target_pos)
        #dq = solver.solve_ik(rob, target_pos, target_orientation)
        else:
            dq = solver.solve_ik_with_reduced_ori_constraint(rob, target_pos)
        data.ctrl[0:7] += dq * model.opt.timestep  # 将增量控制输入应用到前7个执行器上

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
