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
        n_steps=512,
        gamma=0.99,
        learning_rate=3e-05,
        ent_coef=0.001,
        clip_range=0.4,
        n_epochs=5,
        max_grad_norm=5,
        vf_coef=0.5700243050787904,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            #net_arch=dict(pi=[64], vf=[64]),             # Tiny
            #net_arch=dict(pi=[64, 64], vf=[64, 64]),     # Small
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # Medium
        ),
    )
}
