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
        batch_size=512,
        n_steps=2048,
        gamma=0.9,
        learning_rate=0.001,
        ent_coef=0.0003,
        clip_range=0.2,
        n_epochs=20,
        max_grad_norm=1,
        vf_coef=0.505416189583675,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64], vf=[64]), # Tiny net arch, interesting
        ),
    )
}
