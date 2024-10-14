"""
Script containg hyperparameters for PPO training on SmallPO-v0 .
Hyperparameters were found using hyperparameter optimization.
"""

import torch

hyperparams = {
    "SmallPO-v0": dict(
        n_envs=4,
        n_timesteps=500_000,
        policy="MlpPolicy",
        batch_size=512,
        n_steps=1024,
        gamma=0.95,
        learning_rate=0.003,
        ent_coef=0.003,
        clip_range=0.2,
        n_epochs=5,
        max_grad_norm=5,
        vf_coef=0.2743290431835812,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            #net_arch=dict(pi=[64], vf=[64]),             # Tiny
            #net_arch=dict(pi=[64, 64], vf=[64, 64]),     # Small
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # Medium
        ),
    )
}