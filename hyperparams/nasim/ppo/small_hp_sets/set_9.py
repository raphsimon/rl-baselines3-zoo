"""
Script containg hyperparameters for PPO training on Small-v0 .
Hyperparameters were found using hyperparameter optimization.
"""

import torch

hyperparams = {
    "Small-v0": dict(
        n_envs=4,
        n_timesteps=500_000,
        policy="MlpPolicy",
        batch_size=128,
        n_steps=256,
        gamma=0.98,
        learning_rate=3e-05,
        ent_coef=0.0001,
        clip_range=0.2,
        n_epochs=10,
        max_grad_norm=0.3,
        vf_coef=0.934870109311742,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
    )
}
