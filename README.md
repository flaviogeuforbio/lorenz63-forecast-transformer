# Transformer Multi-Step Forecasting on Lorenz-63

Multi-step forecasting of nonlinear dynamical systems across chaotic and non-chaotic regimes.

![forecast gif](plots/true_vs_forecast_chaotic.gif)

<sub>
The forecast shown above is generated using an autoregressive rollout.  
Starting from a fixed input window, the model predicts the next step, which is then fed back as input to iteratively generate a longer trajectory.  
This allows evaluation beyond the fixed training horizon while preserving maximum short-term accuracy at each step.  
The divergence observed at later times is therefore a consequence of intrinsic chaotic sensitivity, not a single-shot long-horizon prediction artifact.
</sub>

---

Predicting the future of a nonlinear system is fundamentally different depending on whether the system is stable or chaotic.  
In chaotic regimes, trajectories **diverge exponentially from infinitesimal perturbations**.  
This project investigates how a Transformer behaves under these fundamentally different dynamical conditions.

---

# What this project does

This repository explores **H-step forecasting** on the Lorenz-63 system:

- Input: a window of past states   
- Output: a multi-step forecast of future states  
- Comparison across:
  - Sub-critical (stable) regime ($\rho < \rho_c$) 
  - Super-critical (chaotic) regime ($\rho > \rho_c$)  

The central question:

> Can a Transformer learn meaningful multi-step dynamics in a chaotic system, where predictability is intrinsically limited?

---

# Key Results

![mse vs horizon](plots/mse_per_step_by_regime.png)

The central metric used to evaluate forecasting performance is the mean squared error (*MSE*) as a function of the forecast horizon. Instead of reporting a single aggregated number, the error is analyzed step-by-step, which makes it possible to observe how predictive accuracy degrades over time and how this degradation depends on the dynamical regime.

In the **sub-critical** (non-chaotic) **regime**, the error remains almost constant across the forecast horizon. This behavior reflects the intrinsic stability of the system: **trajectories tend to converge toward stable attractors**, and small prediction errors do not amplify significantly over time. In this regime, the Transformer maintains accurate multi-step forecasts well beyond the immediate horizon.

In the **super-critical** (chaotic) **regime**, the behavior is qualitatively different. The prediction error increases with forecast step, as expected for a system exhibiting sensitive dependence on initial conditions. However, the growth is structured rather than random: the Transformer consistently outperforms the persistence baseline, and **the predicted trajectories remain confined to the attractor manifold** even when diverging from the exact ground truth trajectory.

The persistence baseline, which simply repeats the last observed state across the forecast horizon, provides a lower bound on trivial continuation. The Transformer surpasses this baseline in both regimes, demonstrating that it learns meaningful short-term dynamics rather than exploiting superficial statistical correlations.

Overall, the results show that the model captures the local flow of the system effectively, while the long-horizon degradation observed in the chaotic regime reflects **intrinsic limits of predictability** rather than model instability.

---

# Why forecasting? (And why not just classification)

The project initially included two auxiliary tasks: regime classification (sub-critical vs super-critical) and regression of the control parameter ρ. These tasks were implemented to evaluate whether complex sequence models were necessary to extract meaningful information from the trajectories.

A straightforward analysis of the dataset, however, revealed that **regime separation is structurally simple**. Even basic window-level statistics, such as the standard deviation of the trajectory within a window, are sufficient to almost perfectly separate the two regimes.

![regime separability](data/plots/meanstd_by_regime.png)

In practice, a lightweight CNN1D architecture already achieves high performance on these tasks (regime classification and rho regression), confirming that they are not dynamically demanding.

*(CNN1D metrics image)*

Forecasting, on the other hand, is fundamentally different. It requires modeling the temporal evolution of the system and capturing its local flow in state space. Unlike classification, which can rely on static distributional differences, **multi-step prediction forces the model to learn how the state transforms over time**.

> For this reason, the focus of the project was shifted from regime/ρ detection to multi-step forecasting, which constitutes the physically meaningful and dynamically non-trivial problem.

---

# Dataset & Sampling Strategy

The dataset is generated from simulated Lorenz-63 trajectories by varying the control parameter \( \rho \) across sub-critical and super-critical regimes. Long trajectories are segmented into fixed-length sliding windows and split into train, validation, and test sets.

A central issue in dynamical systems datasets is the presence of *dead windows*: segments where the trajectory has already collapsed onto the attractor and exhibits very low local variability. Such windows are statistically redundant and can artificially simplify forecasting, especially in sub-critical regimes where convergence to equilibrium is rapid.

To address this, I performed a systematic analysis of window-wise standard deviation as a function of the starting index along each trajectory. This analysis revealed that the early portion of trajectories contains meaningful transient dynamics, while later segments increasingly collapse toward steady attractor behavior.

<p align="center">
  <img src="data/plots/optimal_windows_start.png" width="650">
</p>

Based on this empirical inspection, the following sampling bounds were selected:

- `burn_in ≈ 400`
- `stop_idx ≈ 2000`

These values ensure that windows are extracted from dynamically informative regions while avoiding trivial steady-state segments.

Even simple statistics such as window variance largely separate the two regimes, which explains why classification tasks are comparatively simple.

After applying the selected burn-in and sampling bounds, the fraction of dead windows in each dataset split is:

| Split       | Dead Windows Fraction |
|------------|-----------------------|
| Train      | 0.513                 |
| Validation | 0.534                 |
| Test       | 0.511                 |

These values indicate that approximately half of the windows still lie in low-variance attractor regions, but the sampling procedure prevents the dataset from being dominated by trivial steady states. Forecasting performance therefore reflects genuine dynamical modeling rather than exploitation of collapsed trajectories.



