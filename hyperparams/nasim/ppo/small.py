"""
Script containing default parameters for Small-v0 environment.
No wrappers or anything.
"""


hyperparams = {
    "Small-v0": dict(
        n_envs=4,
        n_timesteps=400_000,
        policy="MlpPolicy",
    )
}
