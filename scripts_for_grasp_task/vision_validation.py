import time
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from grasp_task_rl_config import PandaReachEnv

# 1. 实例化环境
# 必须开启 render_mode="human" 才能看到弹窗窗口
env = PandaReachEnv(render_mode="human")

# 2. 建议加上训练时相同的步数限制，保证行为一致
env = TimeLimit(env, max_episode_steps=200)

# 3. 加载训练好的模型
# 确保文件名和路径与你保存的一致（例如 "ppo_3" 或 "panda_reach_ppo"）
model = PPO.load(
                # "panda_reach_ppo_margin2e-3_50wsteps"
                # "panda_reach_ppo"
                # "mar2e-3_100wsteps"
                "m0_100wsteps"
                 , device="cpu"
                ) 

print("开始测试模型性能...")

# 4. 运行测试循环
obs, info = env.reset()
for i in range(10): # 连续跑 10 个回合
    done = False
    total_reward = 0
    while not done:
        # 使用模型进行推理
        # deterministic=True 表示取消随机性，直接输出模型认为的最优动作
        action, _states = model.predict(obs, deterministic=True)
        
        # 与环境交互
        obs, reward, terminated, truncated, info = env.step(action)

        g_force = 255 - (action[7] + 1.0) / 2.0 * 255.0
        status_str = (
            f"Gripper Force: {g_force:>6.1f} | "
            f"Reward: {reward:>6.2f} | "
            f"Height: {info.get('lift_height', 0):>6.3f}"
        )
        print(f"\r{status_str}", end="", flush=True)
        if terminated or truncated:
            print("\nEpisode finished.") # 换行，防止下一轮覆盖这条信息
            obs, _ = env.reset()
        
        total_reward += reward
        done = terminated or truncated
        
        # 控制显示速度，方便肉眼观察（取决于你的 render_fps）
        time.sleep(0.01) 
        
    print(f"第 {i+1} 回合结束，总得分: {total_reward:.2f}")
    obs, info = env.reset()

env.close()