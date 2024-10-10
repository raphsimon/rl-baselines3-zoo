"""
Script containg hyperparameters for PPO training on SmallPOGen-v0 
environment with StochasticEpisodeStarts.
Hyperparameters were found using hyperparameter optimization.
"""

import torch

hyperparams = {
    "SmallGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=500_000,
        policy="MlpPolicy",
        batch_size=64,
        n_steps=64,
        gamma=0.98,
        learning_rate=0.0003,
        ent_coef=0.01,
        clip_range=0.1,
        n_epochs=5,
        max_grad_norm=2,
        vf_coef=0.1785778246082953,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64], vf=[64]), # Tiny net arch, interesting
        ),
    )
}
