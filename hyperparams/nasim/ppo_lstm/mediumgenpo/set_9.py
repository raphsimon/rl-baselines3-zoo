
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpPolicy",
        batch_size=256,
        n_steps=128,
        gamma=0.95,
        learning_rate=3e-05,
        ent_coef=0.01,
        clip_range=0.1,
        n_epochs=20,
        max_grad_norm=0.6,
        vf_coef=0.6956525211881767,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
    )
}
