# Tracker Evaluation Results

## Setup
- Checkpoint: `legged-loco/logs/rsl_rl/go2_base/001/model_1999.pt`
- Trajectory: `legged-loco/traj.npz` (61 waypoints)
- Seeds: 10  |  Max steps: 2000

## Results
| Metric | Value |
|---|---|
| Completion rate | 10/10 = **1.00** |
| Mean tracking error | 0.0836 m |
| Max tracking error (over all runs) | 0.6579 m |
| Mean completion time | 13.3 s |

## Per-seed
| seed | completed | steps | mean_e_t [m] | max_e_t [m] | terminated_by |
|---|---|---|---|---|---|
| 0 | True | 671 | 0.0533 | 0.1584 | goal_reached |
| 1 | True | 654 | 0.1042 | 0.3965 | goal_reached |
| 2 | True | 635 | 0.0513 | 0.1454 | goal_reached |
| 3 | True | 712 | 0.1338 | 0.6507 | goal_reached |
| 4 | True | 707 | 0.0886 | 0.2660 | goal_reached |
| 5 | True | 564 | 0.1063 | 0.3875 | goal_reached |
| 6 | True | 655 | 0.0787 | 0.5642 | goal_reached |
| 7 | True | 712 | 0.0492 | 0.1465 | goal_reached |
| 8 | True | 662 | 0.0630 | 0.2086 | goal_reached |
| 9 | True | 668 | 0.1070 | 0.6579 | goal_reached |

## Milestone 1 criterion
Completion rate >= 0.9 -> **PASS (1.00)**
