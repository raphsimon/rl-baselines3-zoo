
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpPolicy",
        batch_size=128,
        n_steps=1024,
        gamma=0.98,
        learning_rate=0.001,
        ent_coef=0.003,
        clip_range=0.1,
        n_epochs=5,
        max_grad_norm=1,
        vf_coef=0.85732539698378,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    )
}
