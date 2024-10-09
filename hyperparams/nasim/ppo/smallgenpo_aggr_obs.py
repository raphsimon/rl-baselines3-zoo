"""
Script containing default parameters for SmallGenPO-v0 environment
with aggregated observations.
Two wrappers for this are required.
"""


hyperparams = {
    "SmallGenPO-v0": dict(
        n_envs=4,
        n_timesteps=400_000,
        policy="MlpPolicy",
        env_wrapper=["nasim.envs.wrappers.StochasticEpisodeStarts",
                     "nasim.envs.wrappers.AggregatedObsActionInfoWrapper"],
    )
}