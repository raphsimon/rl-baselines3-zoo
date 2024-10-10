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
        batch_size=256,
        n_steps=512,
        gamma=0.95,
        learning_rate=0.0003,
        ent_coef=1e-05,
        clip_range=0.1,
        n_epochs=20,
        max_grad_norm=0.7,
        vf_coef=0.23080173103198798,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64], vf=[64]), # Tiny net arch, interesting
        ),
    )
}
