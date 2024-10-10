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
        batch_size=128,
        n_steps=2048,
        gamma=0.98,
        learning_rate=0.003,
        ent_coef=0.0003,
        clip_range=0.1,
        n_epochs=10,
        max_grad_norm=2,
        vf_coef=0.8486401609614721,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            #net_arch=dict(pi=[64], vf=[64]),             # Tiny
            #net_arch=dict(pi=[64, 64], vf=[64, 64]),     # Small
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # Medium
        ),
    )
}
