# Transformer Multi-Step Forecasting on Lorenz-63

Multi-step forecasting of nonlinear dynamical systems across chaotic and non-chaotic regimes.

![forecast gif](plots/true_vs_forecast_chaotic.gif)

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

# (B) Key Results

![mse vs horizon](plots/mse_per_step_by_regime.png)

The central metric used to evaluate forecasting performance is the mean squared error (*MSE*) as a function of the forecast horizon. Instead of reporting a single aggregated number, the error is analyzed step-by-step, which makes it possible to observe how predictive accuracy degrades over time and how this degradation depends on the dynamical regime.

In the **sub-critical** (non-chaotic) **regime**, the error remains almost constant across the forecast horizon. This behavior reflects the intrinsic stability of the system: **trajectories tend to converge toward stable attractors**, and small prediction errors do not amplify significantly over time. In this regime, the Transformer maintains accurate multi-step forecasts well beyond the immediate horizon.

In the **super-critical** (chaotic) **regime**, the behavior is qualitatively different. The prediction error increases with forecast step, as expected for a system exhibiting sensitive dependence on initial conditions. However, the growth is structured rather than random: the Transformer consistently outperforms the persistence baseline, and **the predicted trajectories remain confined to the attractor manifold** even when diverging from the exact ground truth trajectory.

The persistence baseline, which simply repeats the last observed state across the forecast horizon, provides a lower bound on trivial continuation. The Transformer surpasses this baseline in both regimes, demonstrating that it learns meaningful short-term dynamics rather than exploiting superficial statistical correlations.

Overall, the results show that the model captures the local flow of the system effectively, while the long-horizon degradation observed in the chaotic regime reflects **intrinsic limits of predictability** rather than model instability.

---

# Why Forecasting? (And Why Not Just Classification)

Initially, the project included:

- Regime classification (sub-critical vs super-critical)
- ρ parameter regression

However, dataset analysis revealed that:

- The two regimes are **easily separable using simple statistics**.
- Even a lightweight CNN1D solves the classification task reliably.

![regime separability](data/plots/meanstd_by_regime.png)

Window-level standard deviation alone largely separates regimes.  
This makes classification/regression tasks structurally simple.

Therefore:

> Forecasting is the physically meaningful task.

It requires modeling temporal evolution, not just detecting statistical signatures.

---

# Dataset & Sampling Strategy

A rigorous dataset pipeline was implemented to avoid trivial artifacts.

## Trajectory Generation

- Numerical integration of Lorenz-63
- ρ sampled in sub- and super-critical intervals
- Fixed σ and β
- Controlled random seeds
- Fixed timestep

## Window Construction

- Input length: `T_in = 128`
- Forecast horizon: `H = 32`
- Sliding window sampling
- Train/Val/Test split

## Burn-in and Transient Removal

Transient collapse toward equilibrium creates "dead windows".

Dead window fractions:

