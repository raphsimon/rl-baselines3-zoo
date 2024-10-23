
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpPolicy",
        batch_size=128,
        n_steps=512,
        gamma=0.95,
        learning_rate=3e-05,
        ent_coef=0.01,
        clip_range=0.1,
        n_epochs=20,
        max_grad_norm=0.5,
        vf_coef=0.8496114059623738,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64], vf=[64]),
        ),
    )
}
