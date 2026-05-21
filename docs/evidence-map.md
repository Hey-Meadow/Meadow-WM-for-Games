# Evidence Map

The website claims should be read from the files under `public/gamml_gym_v1/`.

## Inventory

- JSON artifacts: `57`
- MP4 videos: `68`
- GIF videos: `15`
- Python framework / teacher files: `8`
- Approximate repo artifact size: `118M`

## Main Task Artifacts

| Task | Evidence path | Current wording |
| --- | --- | --- |
| CartPole | `public/gamml_gym_v1/classic_causal_tree/summary.json` | Causal tree solved on recorded classic runs. |
| FlappyBird | `public/gamml_gym_v1/flappybird/summary.json` | Game-like causal-tree success. |
| MinAtar Breakout | `public/gamml_gym_v1/minatar_breakout/summary.json` | Object causal tree works on recorded runs. |
| MountainCar | `public/gamml_gym_v1/mountaincar_energy/summary.json` | Energy-phase tree handles seed, parameter, and terrain variants. |
| Acrobot | `public/gamml_gym_v1/acrobot_energy_causal_tree/summary.json` | Energy causal tree solves recorded seeds. |
| LunarLander | `public/gamml_gym_v1/lunarlander_causal_tree/summary.json` | Successful landing demos, not full benchmark coverage. |
| Pendulum | `public/gamml_gym_v1/pendulum_torque_causal_tree/summary.json` | Recorded-seed torque-control success. |
| MiniGrid DoorKey | `public/gamml_gym_v1/minigrid_doorkey_causal_tree/summary.json` | Symbolic rule-memory causal tree works. |
| Atari Pong | `public/gamml_gym_v1/atari_pong_causal_tree/summary.json` | Pixel object extraction smoke, not solved. |
| BipedalWalker | `public/gamml_gym_v1/bipedalwalker_causal_tree_success/summary.json` | Gait-prior causal tree teacher success. |
| CarRacing | `public/gamml_gym_v1/carracing_per_corner_memory_tree/summary.json` | Racing-line memory causal tree teacher success. |

## Public Wording Rule

Use `artifact viewer`, `recorded-seed`, `Think teacher`, `privileged state`, and `smoke test` where appropriate. Avoid wording that implies a complete benchmark result unless a summary file contains that exact benchmark evidence.
