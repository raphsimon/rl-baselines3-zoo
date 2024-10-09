"""
Script containg hyperparameters for PPO training on SmallPO-v0 
environment with AggregatedObsActionInfoWrapper.
Hyperparameters were found using hyperparameter optimization.
"""

import torch

hyperparams = {
    "SmallPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.AggregatedObsActionInfoWrapper"],
        n_envs=4,
        n_timesteps=400_000,
        policy="MlpPolicy",
        batch_size=64,
        n_steps=512,
        gamma=0.98,
        learning_rate=0.001,
        ent_coef=3e-05,
        clip_range=0.1,
        n_epochs=20,
        max_grad_norm=2,
        vf_coef=0.5342280549586377,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    )
}
