
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpLstmPolicy",
        batch_size=128,
        n_steps=256,
        gamma=0.95,
        learning_rate=0.0001,
        ent_coef=0.001,
        clip_range=0.1,
        n_epochs=20,
        max_grad_norm=0.3,
        vf_coef=0.1230807142810108,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            enable_critic_lstm=False,
            lstm_hidden_size=128,
        ),
    )
}