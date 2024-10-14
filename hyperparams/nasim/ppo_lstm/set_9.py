
import torch

hyperparams = {
    "SmallGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=500000,
        policy="MlpLstmPolicy",
        batch_size=512,
        n_steps=256,
        gamma=0.95,
        learning_rate=0.0003,
        ent_coef=0.03,
        clip_range=0.4,
        n_epochs=10,
        max_grad_norm=0.5,
        vf_coef=0.2442810433991892,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256], vf=[256]),
            enable_critic_lstm=False,
            lstm_hidden_size=256,
        ),
    )
}
