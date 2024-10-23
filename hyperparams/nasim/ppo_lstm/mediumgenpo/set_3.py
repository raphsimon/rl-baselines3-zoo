
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpPolicy",
        batch_size=256,
        n_steps=256,
        gamma=0.99,
        learning_rate=0.001,
        ent_coef=0.01,
        clip_range=0.3,
        n_epochs=20,
        max_grad_norm=0.5,
        vf_coef=0.7212883585088175,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
    )
}
