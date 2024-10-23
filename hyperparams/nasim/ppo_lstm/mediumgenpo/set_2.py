
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpPolicy",
        batch_size=256,
        n_steps=256,
        gamma=0.9,
        learning_rate=3e-05,
        ent_coef=0.03,
        clip_range=0.2,
        n_epochs=20,
        max_grad_norm=2,
        vf_coef=0.270784195381888,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64], vf=[64]),
        ),
    )
}
