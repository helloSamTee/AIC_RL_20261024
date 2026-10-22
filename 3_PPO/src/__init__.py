"""PPO CarRacing workshop package.

A teaching implementation of a Proximal Policy Optimization (PPO) agent that
learns to drive in the Gymnasium ``CarRacing-v3`` environment, which has a
continuous action space and raw 96x96x3 RGB image observations.

The single most important idea this module teaches is that PPO is *on-policy*:
there is NO replay buffer. A short-lived rollout buffer fills to a fixed horizon,
is consumed once by the update, then cleared -- collect, update, clear, repeat.
The code is organized as one small, heavily-commented module per PPO concept
(config, networks, policy, rollout buffer, PPO update, training loop) so a
workshop learner can read each file top-to-bottom and understand one idea at a
time.
"""
