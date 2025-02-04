import rl_zoo3
import rl_zoo3.train
from rl_zoo3.train import train
from sbx import DQN, PPO, SAC, TQC, DroQ


rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["droq"] = DroQ
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["dqn"] = DQN
rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":
    train()

