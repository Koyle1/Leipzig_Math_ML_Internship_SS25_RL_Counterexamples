from gymnasium.envs.registration import register

register(
    id="GraphEnv-v0",
    entry_point="enviroment.graphEnv:GraphConstructionEnv",
)

register(
    id="SeqGraphEnv-v0",
    entry_point="enviroment.seqGraphEnv:SeqGraphConstructionEnv",    
)