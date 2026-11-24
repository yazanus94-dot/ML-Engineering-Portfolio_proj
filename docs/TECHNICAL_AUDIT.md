# Technical audit — ML Engineering Portfolio

## Scope
Audited the five existing subprojects, root CI and documentation without modifying unrelated repositories.

## Initial test baseline (re-run during audit)
| Project | Tests | Result |
|---|---:|---|
| ml-numpy-from-scratch | 20 | pass |
| ml-numerical-debugging-lab | 8 | pass |
| ml-inference-benchmark | 6 | pass |
| ml-pytorch-training | 8 | pass |
| coding-agent-benchmark | 3 | pass |
| **Total** | **45** | pass |

## Findings and actions
| Component | Problem | Severity | Proposed fix | Existing evidence | Verification required |
|---|---|---|---|---|---|
| `nnscratch/losses.py` | softmax could receive malformed labels without explicit validation | medium | validate logits/labels dimensions and ranges | tests pass | `pytest -q` |
| `nnscratch/optim.py` | optimizers lacked validation and type hints | low | add parameter validation and type annotations | tests pass | `pytest -q` |
| Root CI | Node 20 deprecation in checkout/setup actions | low | upgrade to v7 | workflow syntax | GitHub run |
| Root README | no explicit ML researcher role framing | medium | update README and docs | documentation | review |
| Applied research project | no verified public dataset available | blocker | do not invent data; improve existing evidence | blocker documented | explicit approval needed |
| `coding-agent-benchmark` | golden self-check is not an agent score | low | keep human review pending | README already states this | tests pass |

## Security review
No secrets, credentials or patient data were found in tracked files. The `archive/*.zip` files should remain immutable and are not installed by CI.
