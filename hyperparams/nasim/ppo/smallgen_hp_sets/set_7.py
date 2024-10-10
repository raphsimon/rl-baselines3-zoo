"""
Script containg hyperparameters for PPO training on SmallGen-v0 .
Hyperparameters were found using hyperparameter optimization.
"""

import torch

hyperparams = {
    "SmallGen-v0": dict(
        n_envs=4,
        n_timesteps=500_000,
        policy="MlpPolicy",
        batch_size=512,
        n_steps=1024,
        gamma=0.9,
        learning_rate=0.001,
        ent_coef=0.0001,
        clip_range=0.2,
        n_epochs=20,
        max_grad_norm=0.7,
        vf_coef=0.419582318349943,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            #net_arch=dict(pi=[64], vf=[64]),             # Tiny
            #net_arch=dict(pi=[64, 64], vf=[64, 64]),     # Small
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # Medium
        ),
    )
}
