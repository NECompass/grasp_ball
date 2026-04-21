import os
import multiprocessing
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import TimeLimit

from grasp_task_rl_config import PandaReachEnv

# fine tuning 微调
# 目前存在的问题：机械臂反复拿起又放下，推测是机械臂夹爪夹不住，为了保持共线、对称得分才反复上下运动；
# value loss值太大，推测是达到拿到奖励的高度时接触判断有问题（并非）；
# margin配置问题（目前可以通过微调解决）；机械臂前期抓球时的撞球问题（通过加入位移惩罚优化）
class RewardDetailCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardDetailCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            for key in info.keys():
                if key.startswith("charts/"):
                    self.logger.record(key[7:], info[key])
        return True

def make_env(rank, seed=0):
    def _init():
        env = PandaReachEnv(render_mode=None)
        env = TimeLimit(env, max_episode_steps=150)
        env = Monitor(env)
        env.reset(seed=seed+rank)
        return env
    
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver', force=True)
    num_cpu = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # env = DummyVecEnv([make_env(0)])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_path = os.path.join(parent_dir, 'panda_reach_ppo_margin2e-3_50wsteps.zip')
    model = PPO.load(model_path, env=env, device="cpu")

    reward_callback = RewardDetailCallback()

    try:
        model.learn(total_timesteps=500000, reset_num_timesteps=False, callback = reward_callback)
        model.save("m0_100wsteps")
    except Exception as e:
        print(f"训练中断：{e}")
    finally:
        env.close()