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
        n_steps=1024,
        gamma=0.99,
        learning_rate=0.0001,
        ent_coef=0.003,
        clip_range=0.2,
        n_epochs=20,
        max_grad_norm=1,
        vf_coef=0.1161590899607925,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    )
}
