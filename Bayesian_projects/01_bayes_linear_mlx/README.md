# Bayesian Linear Model (MLX)

A single Bayesian linear layer trained on synthetic air pollution → brain health data.

## What it does

Learns a weight distribution (not a point estimate) using variational inference. Instead of `w`, the model maintains `w_mu` and `w_rho`, sampling a weight each forward pass via the reparameterization trick.

**Simulated relationship:** `brain_health = -1.5 * pollution + 0.5 + noise`

## Files

| File | Purpose |
|------|---------|
| `bayes_model.py` | `BayesianLayer` class + ELBO loss |
| `training.py` | Training loop (500 epochs, Adam, `mx.compile`) |
| `sample_data.py` | 100-sample synthetic dataset |

## Run

```bash
cd Bayesian_projects/01_bayes_linear_mlx
python training.py
```

## Key concepts

- **w_mu / w_rho** — mean and log-scale of the weight distribution
- **Softplus** — keeps sigma positive: `σ = log(1 + exp(ρ))`
- **ELBO loss** — prediction error + KL penalty (complexity cost / 100)
- **`mx.compile` with state** — required so MLX's lazy graph actually updates weights
