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
        batch_size=256,
        n_steps=64,
        gamma=0.9,
        learning_rate=0.0001,
        ent_coef=0.0003,
        clip_range=0.4,
        n_epochs=20,
        max_grad_norm=0.5,
        vf_coef=0.8208833338711006,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64], vf=[64]), # Tiny net arch, interesting
        ),
    )
}
