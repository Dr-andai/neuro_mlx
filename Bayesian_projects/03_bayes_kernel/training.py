import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import rff_bkmr
import sample_data

n_rff = 200
lengthscale = 1.5

rff = rff_bkmr.RandomFourierFeatures(
    input_dim=sample_data.M,
    n_features=n_rff,
    lengthscale=lengthscale,
    kernel_variance=1.0,
    seed=42,
)

phi_z_train = rff(sample_data.Z_train)
phi_z_test = rff(sample_data.Z_test)

model = rff_bkmr.BKMR_RFF(n_rff=phi_z_train.shape[-1], n_covariates=sample_data.X_train.shape[-1])

optimizer = optim.Adam(learning_rate=0.005)


def step(phi_z, Y, X):
    loss, grads = nn.value_and_grad(model, rff_bkmr.elbo_loss)(model, phi_z, Y, X)
    optimizer.update(model, grads)
    return loss


state = [model.state, optimizer.state]
compiled_step = mx.compile(step, inputs=state, outputs=state)

n_epochs = 2000
best_rmse = float("inf")

for epoch in range(n_epochs):
    loss = compiled_step(phi_z_train, sample_data.Y_train, sample_data.X_train)
    mx.eval(model.parameters(), optimizer.state, loss)

    if epoch % 400 == 0:
        train_pred = model.predict(phi_z_train, sample_data.X_train)
        train_rmse = mx.sqrt(mx.mean(mx.square(train_pred - sample_data.Y_train.squeeze()))).item()
        test_pred = model.predict(phi_z_test, sample_data.X_test)
        test_rmse = mx.sqrt(mx.mean(mx.square(test_pred - sample_data.Y_test.squeeze()))).item()
        sigma = mx.exp(model.log_sigma).item()
        print(f"Epoch {epoch:4d} | ELBO: {loss.item():.4f} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | sigma: {sigma:.4f}")
        if test_rmse < best_rmse:
            best_rmse = test_rmse

print(f"\n--- Training Complete (best test RMSE: {best_rmse:.4f}) ---")

pred_mean, pred_std = model.predict_with_uncertainty(phi_z_test, sample_data.X_test, n_samples=100)
final_rmse = mx.sqrt(mx.mean(mx.square(pred_mean - sample_data.Y_test.squeeze()))).item()
print(f"Test RMSE (posterior mean samples): {final_rmse:.4f}")
print(f"Avg posterior std: {mx.mean(pred_std).item():.4f}")

alpha_sigma = mx.log(1.0 + mx.exp(model.alpha_rho))
print(f"alpha_sigma: mean={mx.mean(alpha_sigma).item():.4f}, median={mx.median(alpha_sigma).item():.4f}")

h_true_test = sample_data.h_test
h_est_test = phi_z_test @ model.alpha_mu
h_rmse = mx.sqrt(mx.mean(mx.square(h_est_test - h_true_test))).item()
print(f"h(z) estimation RMSE (test): {h_rmse:.4f}")
