from gymnasium.envs.registration import register

register(
    id="conjecture_1",
    entry_point="environment.env:GraphNodeBuildEnv",
)
