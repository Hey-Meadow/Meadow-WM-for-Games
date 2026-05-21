# Framework

Meadow WM for Games uses games as compact control laboratories for the same Meadow world-model loop used in robotics and physical systems.

## Core Loop

```text
GameSpec
  -> GoalSpec
  -> Think causal tree
  -> Evidence artifacts
  -> Reaction distillation
```

## Components

- `GameSpec` describes the environment state, action space, physical constants, clone/restore capability, privileged state access if any, and termination conditions.
- `GoalSpec` rewrites game score into a checkable success condition, such as safe landing, stable balance, reaching a flag, swing-up, door-key completion, or lap completion.
- `Think causal tree` expands candidate action chains or reusable primitives, scores future branches, records selected chains and near misses, and preserves failure reasons.
- `Evidence artifacts` are the JSON summaries, plan traces, and MP4/GIF rollouts used by the public site.
- `Reaction distillation` is the downstream target: selected causal chains can become supervised action chunks or low-latency policy training rows.

## Honest Boundaries

- The v1 site is not a full OpenAI Gym benchmark claim.
- Some teachers use simulator state, Gymnasium clone/restore, or privileged environment fields.
- Recorded-seed success is separated from cross-seed robustness.
- Pong is only a smoke test.
- LunarLander is a landing demo set, not a 100-episode benchmark.
