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
        batch_size=256,
        n_steps=512,
        gamma=0.98,
        learning_rate=0.0003,
        ent_coef=0.003,
        clip_range=0.2,
        n_epochs=20,
        max_grad_norm=0.8,
        vf_coef=0.19977617271169568,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    )
}
