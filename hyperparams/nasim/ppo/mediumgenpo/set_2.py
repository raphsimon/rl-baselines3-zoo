
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpPolicy",
        batch_size=128,
        n_steps=1024,
        gamma=0.9,
        learning_rate=0.0003,
        ent_coef=0.0001,
        clip_range=0.2,
        n_epochs=5,
        max_grad_norm=2,
        vf_coef=0.255176225767017,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64], vf=[64]),
        ),
    )
}
