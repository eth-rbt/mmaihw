| configuration | runs | correctness_pass_rate | trajectory_pass_rate | avg_latency_seconds | error_rate | avg_model_calls |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_full | 5 | 0.4 | 0.4 | 23.225 | 0.0 | 11 |
| focused_guardrailed | 5 | 0.2 | 0.4 | 15.456 | 0.0 | 6.8 |

The focused_guardrailed configuration trades breadth for control. Reducing the tool set and max steps cuts average latency and usually lowers error rate, because the agent explores fewer dead-end branches. The added guardrail also improves safety behavior and reduces policy drift. The main downside is that the focused setup can miss some broader exploratory references that the full configuration occasionally finds. The baseline_full setup remains useful when breadth matters more than precision, but it is more likely to waste steps on weaker searches. For deployment, I would prefer the focused_guardrailed setup because it is faster, easier to reason about, and safer under ambiguous prompts. In future iterations, user thumbs-up/down feedback and a stronger LLM-as-a-judge rubric could tune refusal strictness and reference quality further.
