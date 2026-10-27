# Batch float64 finish on neos-3627168-kasai

32 strong-branching LPs, streamed through 8 slots, target tolerance 1e-4, device budget
`max_iters=60000`, 2-core cloud CPU.

| setting | time | optimal | worst obj err vs HiGHS | mean iterations per LP | of which CPU float64 finish |
|---|---|---|---|---|---|
| CPU float64 only | 55.2 s | 32/32 | 3.3e-07 | 28,856 | 0 |
| emulated Trainium, float32 only | 454.0 s | 12/32 | 7.3e-05 | 52,822 | 0 |
| emulated Trainium + float64 finish | 465.1 s | 32/32 | 7.3e-05 | 60,074 | 7,252 (12%) |

Most LPs reached the device budget rather than tripping the stall rule. kasai plateaus for long
stretches in float64 too (one branch: 25k float32 vs 27k float64 iterations, same flat-then-drop
shape), so it's slow rather than precision-limited, and the conservative stall rule correctly
doesn't fire mid-plateau.
