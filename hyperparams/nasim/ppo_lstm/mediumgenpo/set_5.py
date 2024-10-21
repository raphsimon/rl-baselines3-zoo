
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpLstmPolicy",
        batch_size=256,
        n_steps=256,
        gamma=0.95,
        learning_rate=3e-05,
        ent_coef=0.0003,
        clip_range=0.1,
        n_epochs=20,
        max_grad_norm=0.6,
        vf_coef=0.3790306558540846,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64], vf=[64]),
            enable_critic_lstm=False,
            lstm_hidden_size=128,
        ),
    )
}
