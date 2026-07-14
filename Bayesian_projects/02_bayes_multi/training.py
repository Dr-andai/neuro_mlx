import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import bayes_multi
import ingest_data

# get model
in_dim      = ingest_data.X_train.shape[1]
hidden_dim  = 64
num_classes = 4
model = bayes_multi.BayesianClassificationMLP(in_dim, hidden_dim, num_classes)

# get data
X_train = ingest_data.X_train
y_train = ingest_data.y_train
X_test  = ingest_data.X_test
y_test  = ingest_data.y_test

# set optimizer
optimizer = optim.Adam(learning_rate=1e-3)

N           = X_train.shape[0]
BATCH_SIZE  = 256
num_batches = max(1, N // BATCH_SIZE)


# core training step — called once per mini-batch
def training_step(x_batch, y_batch):
    # nn.value_and_grad computes the ELBO loss and all gradients in one pass
    loss_value, gradients = nn.value_and_grad(model, bayes_multi.elbo_loss)(
        model, x_batch, y_batch, num_batches
    )
    optimizer.update(model, gradients)
    return loss_value


# Compile the step with explicit state tracking so weight updates persist.
# Without inputs/outputs, MLX treats model and optimizer state as frozen
# constants — parameters would appear to change inside the step but revert
# on the next call.
state = [model.state, optimizer.state]
compiled_step = mx.compile(training_step, inputs=state, outputs=state)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

EPOCHS = 20

for epoch in range(EPOCHS):
    # Shuffle each epoch so mini-batches sample the data differently
    perm = np.random.permutation(N)
    epoch_loss = 0.0

    for i in range(num_batches):
        idx   = mx.array(perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])
        x_b   = X_train[idx]
        y_b   = y_train[idx]

        loss  = compiled_step(x_b, y_b)

        # MLX is lazy — mx.eval forces queued computations to run and keeps
        # memory from growing unboundedly across batches.
        mx.eval(model.parameters(), optimizer.state, loss)

        epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:>3}/{EPOCHS}  avg_loss={epoch_loss / num_batches:.4f}")


print("\n--- Training Complete! ---")


# ---------------------------------------------------------------------------
# Monte Carlo inference with uncertainty
# ---------------------------------------------------------------------------
# Run num_samples forward passes; each pass draws a fresh set of weights.
# Averaging the softmax outputs approximates the Bayesian posterior predictive.
# Std across samples = epistemic uncertainty — "how unsure is the model?"

NUM_SAMPLES = 50
all_probs   = []

for _ in range(NUM_SAMPLES):
    logits = model(X_test)
    probs  = mx.softmax(logits, axis=-1)
    all_probs.append(probs)
    mx.eval(probs)

stacked     = mx.stack(all_probs, axis=0)               # (S, N_test, classes)
mean_probs  = mx.mean(stacked, axis=0)                   # (N_test, classes)
uncertainty = mx.mean(mx.std(stacked, axis=0), axis=-1)  # (N_test,)
preds       = mx.argmax(mean_probs, axis=-1)

correct = mx.sum(preds == y_test).item()
total   = y_test.shape[0]
print(f"\nTest accuracy: {correct}/{total}  ({100 * correct / total:.1f}%)")

print("\nSample predictions (first 8):")
for i in range(8):
    print(
        f"  [{i}]  pred={preds[i].item()}  "
        f"true={y_test[i].item()}  "
        f"uncertainty={uncertainty[i].item():.4f}"
    )
