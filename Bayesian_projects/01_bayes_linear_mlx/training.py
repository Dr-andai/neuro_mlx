import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
import os
import bayes_model
import sample_data

# get model
model = bayes_model.BayesianLayer()

# get data
X_data = sample_data.X
Y_data = sample_data.Y

# set optimizer
optimizer = optim.Adam(learning_rate=0.05)


# core training
def training_step(X_batch, Y_batch):
    # value and grad calculates the ELBO and mathematical direct
    # need to improve our dials (w_mu, w_rho, etc)
    loss_value, gradients = nn.value_and_grad(model,bayes_model.beginner_elbo_loss)(X_batch, Y_batch, model)

    optimizer.update(model, gradients)
    return loss_value

# ERROR: mx.compile(training_step) without state tracking.
# model.state and optimizer.state are mutated inside training_step,
# but without inputs/outputs the compiled graph treats them as
# frozen constants — weights never update across calls.
state = [model.state, optimizer.state]
compiled_step = mx.compile(training_step, inputs=state, outputs=state)


for epoch in range(500):
    loss = compiled_step(X_data, Y_data)
    # ERROR: mx.eval() was missing after each step.
    # MLX is lazy — computations queue up without executing.
    # Without this, the graph grows unboundedly and model
    # parameter updates are never written to memory.
    mx.eval(model.parameters(), optimizer.state, loss)

    if epoch % 100 == 0:
        w_sigma = mx.log(1.0 + mx.exp(model.w_rho)).item()
        print(f"Step{epoch:3d} | ELBO Loss: {loss.item():.4f} |"
              f" Weight Estimate (μ): {model.w_mu.item():.2f} |"
              f" Weight Uncertainty (σ): {w_sigma:.2f}")
        
print("\n--- Training Complete! ---")

