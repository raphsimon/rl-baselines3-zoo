"""
Script containing default parameters for SmallGen-v0 environment.
This means this is the stochastic environment, but fully observable.
"""


hyperparams = {
    "SmallPO-v0": dict(
        n_envs=4,
        n_timesteps=400_000,
        policy="MlpPolicy",
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts"],
    )
}