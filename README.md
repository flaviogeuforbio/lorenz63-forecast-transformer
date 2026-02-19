# Transformer Multi-Step Forecasting on Lorenz-63

Multi-step forecasting of nonlinear dynamical systems across chaotic and non-chaotic regimes.

![forecast gif](plots/true_vs_forecast_chaotic.gif)

Predicting the future of a nonlinear system is fundamentally different depending on whether the system is stable or chaotic.  
In chaotic regimes, trajectories **diverge exponentially from infinitesimal perturbations**.  
This project investigates how a Transformer behaves under these fundamentally different dynamical conditions.

---

# (A) What this project does

This repository explores **H-step forecasting** on the Lorenz-63 system:

- Input: a window of past states  
- Output: a multi-step forecast of future states  
- Comparison across:
  - Sub-critical (stable) regime  
  - Super-critical (chaotic) regime  

The central question:

> Can a Transformer learn meaningful multi-step dynamics in a chaotic system, where predictability is intrinsically limited?

---

# (B) Key Results

![mse vs horizon](plots/mse_per_step_by_regime.png)

### Observations

- In the **non-chaotic regime**, forecast error remains nearly flat across horizon.
- In the **chaotic regime**, error grows with forecast step (as expected).
- The Transformer significantly outperforms a **persistence baseline**.
- Divergence in the chaotic regime remains dynamically coherent (attractor switching rather than random drift).

This confirms:

- The model captures short-term dynamics meaningfully.
- Long-horizon degradation reflects intrinsic system instability, not model collapse.

---

# (C) Why Forecasting? (And Why Not Just Classification)

Initially, the project included:

- Regime classification (sub-critical vs super-critical)
- ρ parameter regression

However, dataset analysis revealed that:

- The two regimes are **easily separable using simple statistics**.
- Even a lightweight CNN1D solves the classification task reliably.

![regime separability](plots/regime_histogram.png)

Window-level standard deviation alone largely separates regimes.  
This makes classification/regression tasks structurally simple.

Therefore:

> Forecasting is the physically meaningful task.

It requires modeling temporal evolution, not just detecting statistical signatures.

---

# (D) Dataset & Sampling Strategy

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

