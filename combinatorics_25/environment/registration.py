from gymnasium.envs.registration import register

register(
    id="Conjuncture2-GraphEnv-v0",
    entry_point="environment.graphEnvConjuncture2:GraphConstructionEnv",
)

register(
    id="Conjuncture2-SeqGraphEnv-v0",
    entry_point="environment.seqGraphEnvConjuncture2:SeqGraphConstructionEnv",    
)

register(
    id="Conjecture1-GraphEnv-v0",
    entry_point="environment.graphEnvConjuncture1:GraphConstructionEnv",
)