
import torch

hyperparams = {
    "SmallGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=500000,
        policy="MlpLstmPolicy",
        batch_size=256,
        n_steps=2048,
        gamma=0.98,
        learning_rate=0.001,
        ent_coef=1e-05,
        clip_range=0.4,
        n_epochs=5,
        max_grad_norm=0.8,
        vf_coef=0.4305415136062333,
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[256], vf=[256]),
            enable_critic_lstm=False,
            lstm_hidden_size=64,
        ),
    )
}
