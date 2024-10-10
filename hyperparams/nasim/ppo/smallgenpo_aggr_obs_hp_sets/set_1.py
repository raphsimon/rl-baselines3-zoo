"""
Script containg hyperparameters for PPO training on SmallPO-v0 
environment with AggregatedObsActionInfoWrapper.
Hyperparameters were found using hyperparameter optimization.
"""

import torch

hyperparams = {
    "SmallGenPO-v0": dict(
        n_envs=4,
        n_timesteps=600_000,
        policy="MlpPolicy",
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",
                     "nasim.envs.wrappers.AggregatedObsActionInfoWrapper"],
        batch_size=64,
        n_steps=512,
        gamma=0.98,
        learning_rate=3e-05,
        ent_coef=0.0003,
        clip_range=0.2,
        n_epochs=10,
        max_grad_norm=5,
        vf_coef=0.9011949378333878,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            #net_arch=dict(pi=[64], vf=[64]),             # Tiny
            net_arch=dict(pi=[64, 64], vf=[64, 64]),     # Small
            #net_arch=dict(pi=[128, 128], vf=[128, 128]), # Medium
        ),
    )
}
