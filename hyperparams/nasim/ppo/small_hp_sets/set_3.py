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
        gamma=0.95,
        learning_rate=0.0003,
        ent_coef=3e-05,
        clip_range=0.1,
        n_epochs=5,
        max_grad_norm=5,
        vf_coef=0.9149914735321832,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64], vf=[64]), # Tiny net arch, interesting
        ),
    )
}
