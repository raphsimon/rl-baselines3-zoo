
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpPolicy",
        batch_size=64,
        n_steps=64,
        gamma=0.99,
        learning_rate=3e-05,
        ent_coef=0.03,
        clip_range=0.4,
        n_epochs=5,
        max_grad_norm=0.6,
        vf_coef=0.997797131747175,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    )
}
