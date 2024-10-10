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
        batch_size=128,
        n_steps=32,
        gamma=0.99,
        learning_rate=0.0001,
        ent_coef=3e-05,
        clip_range=0.4,
        n_epochs=20,
        max_grad_norm=2,
        vf_coef=0.42413613567086417,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            #net_arch=dict(pi=[64], vf=[64]),             # Tiny
            net_arch=dict(pi=[64, 64], vf=[64, 64]),     # Small
            #net_arch=dict(pi=[128, 128], vf=[128, 128]), # Medium
        ),
    )
}
