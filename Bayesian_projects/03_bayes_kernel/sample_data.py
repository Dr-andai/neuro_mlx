import mlx.core as mx

mx.random.seed(42)

n = 400
M = 5

Z = mx.random.uniform(-2.0, 2.0, (n, M))

z0 = Z[:, 0]
z1 = Z[:, 1]
z2 = Z[:, 2]
z3 = Z[:, 3]
z4 = Z[:, 4]

h = mx.sin(z0) + 0.5 * mx.square(z1) + 0.3 * z2 * z3 + 0.2 * mx.exp(-mx.square(z4))
h = h - mx.mean(h)

X_covar = mx.random.normal((n, 1))

noise = mx.random.normal((n,)) * 0.2
Y = h + X_covar.squeeze() * 1.0 + noise
Y = Y.reshape(-1, 1)

n_train = 300
Z_train = Z[:n_train]
Z_test = Z[n_train:]
h_train = h[:n_train]
h_test = h[n_train:]
X_train = X_covar[:n_train]
X_test = X_covar[n_train:]
Y_train = Y[:n_train]
Y_test = Y[n_train:]
