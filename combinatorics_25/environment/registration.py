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

register(
    id="Conjecture1-GraphEnvPE-v0",
    entry_point="environment.graphEnvConjuncture1PE:GraphConstructionEnv",
)

register(
    id="Conjecture1-GraphEnvOHE-v0",
    entry_point="environment.graphEnvConjuncture1OHE:GraphConstructionEnv",
)

register(
    id="c1-graphNodeBuildEnv-v0",
    entry_point="environment.graphNodeBuildEnv:GraphNodeBuildEnv",
)

register(
    id="c1-graphNodeBuildPE-v0",
    entry_point="environment.graphNodeBuildPE:GraphNodeBuildEnv",
)

register(
    id="c1-graphNodeBuildPEseq-v0",
    entry_point="environment.graphNodeBuildPEseq:GraphNodeBuildEnv",
)

register(
    id="c1-graphNodeBuildPEseqB-v0",
    entry_point="environment.graphNodeBuildPEseqBoundary:GraphNodeBuildEnv",
)

register(
    id="c1-graphNodeBuildPEseqBRONE-v0",
    entry_point="environment.graphNodeBuildPEseqBoundaryNewRewardOnlyOnNewEdgeAndNode:GraphNodeBuildEnv",
)

register(
    id="c1-graphNodeBuildPEseqBRONN-v0",
    entry_point="environment.graphNodeBuildPEseqBoundaryNewRewardOnlyOnNewNode:GraphNodeBuildEnv",
)

register(
    id="c2-graphNodeBuildPEseqB-v0",
    entry_point="environment.graphNodeBuildPEseqBoundary_C2:GraphNodeBuildEnv",
)