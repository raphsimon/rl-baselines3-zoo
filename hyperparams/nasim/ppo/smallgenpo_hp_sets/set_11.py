"""
Script containg hyperparameters for PPO training on SmallPOGen-v0 
environment with StochasticEpisodeStarts.
Hyperparameters were found using hyperparameter optimization.
"""

import torch

hyperparams = {
    "SmallGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=500_000,
        policy="MlpPolicy",
        batch_size=256,
        n_steps=2048,
        gamma=0.95,
        learning_rate=0.001,
        ent_coef=1e-05,
        clip_range=0.2,
        n_epochs=20,
        max_grad_norm=0.9,
        vf_coef=0.5951807906193154,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64, 64], vf=[64, 64]), # Tiny net arch, interesting
        ),
    )
}
