# Results

This page reports what seems to work, what broke along the way, and what would be worth improving next.

## Successes

The project already does a few things reasonably well:

- it trains one model across a family of double-pendulum parameter settings
- it uses a mechanically structured architecture instead of direct black-box acceleration regression
- it produces usable held-out rollouts
- it supports some manual out-of-distribution parameter tests
- it includes diagnostics that help inspect the learned structure, not just the final trajectories

Example results:

### Rollout Predictions
These are examples of rollout prediction from the test set; the parameters of the pendula (blob masses and rod lengths) were not seen during training.

`../results/rollouts/model_T512_20260317_133032_energy_test_0.png`
`../results/rollouts/model_T512_20260317_133032_energy_test_2.png`
`../results/rollouts/model_T512_20260317_133032_energy_test_8.png`


### Energy Tests
The energy loss (which should be 0 in perfect predictions) remains at or below 1% for most if not all the cases. Please note that the vertical axis is already scaled as a percentage value. These refers to the same tests above in 'Rollout Predictions'.

`../results/rollouts/model_T512_20260317_133032_energy_test_0.png`
`../results/rollouts/model_T512_20260317_133032_energy_test_2.png`
`../results/rollouts/model_T512_20260317_133032_energy_test_8.png`

The potential and kinetic energy representation that the model predicts are ok, but you can see how the kinetic energy predictions starts being qualitatively different from the truth when approaching the edges, because of lack of training data in the region.

`../results/kinpot_decomposition/model_T512_20260317_133032_kpd_case_0_qd1.png`

`../results/kinpot_decomposition/model_T512_20260317_133032_kpd_case_1_qd1.png`

### Out of Distribution Predictions
The tests included some parameter and initial condition pairs that are outside of the ranges of values used in training. That means positions, velocities, masses and lengths that were outside the ranges seen in training.

`../results/ood_tests/model_T512_20260317_133032_ood_0.png`
`../results/ood_tests/model_T512_20260317_133032_ood_1.png`
`../results/ood_tests/model_T512_20260317_133032_ood_2.png`

## Failures

The project also ran into plenty of failure modes on the way:

- unstable or meaningless matrix structure before the kinetic branch was constrained properly
- conditioning choices that looked reasonable and trained poorly
- sensitivity to parameter sampling and target quality

That history is important because it explains why the current architecture is shaped the way it is.

### Out of Distribution Failures
When the parameters and/or initial conditions fall too far from the training distribution, the error starts accumulating and because of the high-nonlinearity, results drastically diverge. In this case below, the second blob of the ground truth falls short of a full swing at the very beginning, while the model predicts the full rotation and from there, the trajectories become completely different. 

`../results/ood_tests/model_T512_20260317_133032_ood_3.png`

`../results/sample_viz/ood_comparison_3.gif`



## Current Limitations

The main limitations are:

- the implementation is specialized to the 2-DoF double pendulum
- the learned energy story is in normalized coordinates, not fully physical coordinates
- the current energy-conservation regularizer assumes one trajectory chunk per batch
- several workflows are still driven by script constants and hard-coded examples

## What To Improve Next

If this project were pushed further, the most useful next steps would probably be:

1. make the structured kinetic construction less hard-coded to the 2-DoF case
2. generalize training and evaluation so fewer settings live directly inside scripts
3. make the energy-conservation regularizer explicitly trajectory-local for more flexible batching
4. expand training to larger traiing distribution parameters and initial conditions
5. expand evaluation beyond a few held-out and manual OOD examples
6. keep tightening the docs as the repo grows

## How To Read This Repo

This repo is best read as:

- a focused implementation
- a technically grounded project
- a build that got better by debugging real mistakes

It is not best read as:

- a general mechanics framework
- a benchmark-heavy paper claim
- a final solution to Lagrangian learning

That narrower framing is what makes the project feel solid rather than overclaimed.
