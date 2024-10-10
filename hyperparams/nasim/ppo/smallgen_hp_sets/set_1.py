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
        n_steps=512,
        gamma=0.99,
        learning_rate=0.001,
        ent_coef=0.01,
        clip_range=0.1,
        n_epochs=20,
        max_grad_norm=0.5,
        vf_coef=0.05526231493940936,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            #net_arch=dict(pi=[64], vf=[64]),             # Tiny
            #net_arch=dict(pi=[64, 64], vf=[64, 64]),     # Small
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # Medium
        ),
    )
}
