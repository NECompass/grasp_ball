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

    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps = 1024,
        batch_size=64,
        verbose=1, 
        device = "cpu",
        ent_coef=0.01,
        tensorboard_log = "./ppo_panda_tensorboard"
    )

    reward_callback = RewardDetailCallback()

    try:
        model.learn(total_timesteps=300000, callback = reward_callback)
        model.save("panda_reach_ppo_margin2e-3_50wsteps")
    except Exception as e:
        print(f"训练中断：{e}")
    finally:
        env.close()