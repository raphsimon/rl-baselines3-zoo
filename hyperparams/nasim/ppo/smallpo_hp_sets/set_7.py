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
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        learning_rate=0.0001,
        ent_coef=0.01,
        clip_range=0.3,
        n_epochs=5,
        max_grad_norm=0.9,
        vf_coef=0.6315711651895585,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            #net_arch=dict(pi=[64], vf=[64]),             # Tiny
            #net_arch=dict(pi=[64, 64], vf=[64, 64]),     # Small
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # Medium
        ),
    )
}
