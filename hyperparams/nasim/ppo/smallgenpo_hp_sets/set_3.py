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
        batch_size=128, 
        n_steps=256, 
        gamma=0.98, 
        learning_rate=0.003, 
        ent_coef=0.0003, 
        clip_range=0.2, 
        n_epochs=20, 
        max_grad_norm=0.6, 
        vf_coef=0.9991311790345467, 
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64], vf=[64]), # Tiny net arch, interesting
        ),
    )
}
