
import torch

hyperparams = {
    "MediumGenPO-v0": dict(
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",],
        n_envs=4,
        n_timesteps=5000000,
        policy="MlpLstmPolicy",
        batch_size=256,
        n_steps=1024,
        gamma=0.9,
        learning_rate=0.0001,
        ent_coef=0.03,
        clip_range=0.4,
        n_epochs=20,
        max_grad_norm=2,
        vf_coef=0.5077464111576053,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64], vf=[64]),
            enable_critic_lstm=False,
            lstm_hidden_size=256,
        ),
    )
}
