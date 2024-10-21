
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpPolicy",
        batch_size=256,
        n_steps=1024,
        gamma=0.9,
        learning_rate=3e-05,
        ent_coef=0.003,
        clip_range=0.1,
        n_epochs=5,
        max_grad_norm=0.5,
        vf_coef=0.624266235300737,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64], vf=[64]),
        ),
    )
}
