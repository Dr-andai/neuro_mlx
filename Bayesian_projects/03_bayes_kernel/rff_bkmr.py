import mlx.core as mx
import mlx.nn as nn


class RandomFourierFeatures:
    def __init__(self, input_dim, n_features=100, lengthscale=1.0, kernel_variance=1.0, seed=42):
        self.input_dim = input_dim
        self.n_features = n_features

        mx.random.seed(seed)
        omega = mx.random.normal((input_dim, n_features)) / lengthscale
        bias = mx.random.uniform(0, 2 * mx.pi, (n_features,))

        self.omega = omega
        self.bias = bias
        self.scale = mx.sqrt(2.0 * kernel_variance / n_features)

    def __call__(self, x):
        projection = x @ self.omega + self.bias
        return self.scale * mx.cos(projection)


class BKMR_RFF(nn.Module):
    def __init__(self, n_rff, n_covariates=0):
        super().__init__()
        self.n_rff = n_rff
        self.n_covariates = n_covariates

        bound = 1.0 / mx.sqrt(mx.array(float(n_rff)))
        self.alpha_mu = mx.random.uniform(-bound, bound, (n_rff,))
        self.alpha_rho = mx.full((n_rff,), -3.0)

        if n_covariates > 0:
            bound_cov = 1.0 / mx.sqrt(mx.array(float(n_covariates)))
            self.beta_mu = mx.random.uniform(-bound_cov, bound_cov, (n_covariates,))
            self.beta_rho = mx.full((n_covariates,), -3.0)

        self.log_sigma = mx.zeros((1,))
        self.current_kl_alpha = mx.array(0.0)
        if n_covariates > 0:
            self.current_kl_beta = mx.array(0.0)

    def __call__(self, phi_z, X=None, sample=True):
        alpha_sigma = mx.log(1.0 + mx.exp(self.alpha_rho))
        self.current_kl_alpha = self._kl_gaussian(self.alpha_mu, alpha_sigma, 0.0, 1.0)

        if sample:
            eps_alpha = mx.random.normal(self.alpha_mu.shape)
            alpha = self.alpha_mu + alpha_sigma * eps_alpha
        else:
            alpha = self.alpha_mu
        pred = phi_z @ alpha

        if X is not None and self.n_covariates > 0:
            beta_sigma = mx.log(1.0 + mx.exp(self.beta_rho))
            self.current_kl_beta = self._kl_gaussian(self.beta_mu, beta_sigma, 0.0, 1.0)
            if sample:
                eps_beta = mx.random.normal(self.beta_mu.shape)
                beta = self.beta_mu + beta_sigma * eps_beta
            else:
                beta = self.beta_mu
            pred = pred + X @ beta

        return pred

    def predict(self, phi_z, X=None):
        pred = phi_z @ self.alpha_mu
        if X is not None and self.n_covariates > 0:
            pred = pred + X @ self.beta_mu
        return pred

    def predict_with_uncertainty(self, phi_z, X=None, n_samples=100):
        preds = []
        for _ in range(n_samples):
            alpha_sigma = mx.log(1.0 + mx.exp(self.alpha_rho))
            eps_alpha = mx.random.normal(self.alpha_mu.shape)
            alpha = self.alpha_mu + alpha_sigma * eps_alpha
            pred = phi_z @ alpha
            if X is not None and self.n_covariates > 0:
                beta_sigma = mx.log(1.0 + mx.exp(self.beta_rho))
                eps_beta = mx.random.normal(self.beta_mu.shape)
                beta = self.beta_mu + beta_sigma * eps_beta
                pred = pred + X @ beta
            preds.append(pred)
        all_preds = mx.stack(preds, axis=0)
        return mx.mean(all_preds, axis=0), mx.std(all_preds, axis=0)

    def total_kl(self):
        kl = self.current_kl_alpha
        if self.n_covariates > 0:
            kl = kl + self.current_kl_beta
        return kl

    def _kl_gaussian(self, mu_q, sigma_q, mu_p=0.0, sigma_p=1.0):
        return 0.5 * mx.sum(
            -1.0
            - mx.log(mx.square(sigma_q) / mx.square(sigma_p))
            + mx.square(sigma_q) / mx.square(sigma_p)
            + mx.square(mu_q - mu_p) / mx.square(sigma_p)
        )


def elbo_loss(model, phi_z, Y, X):
    pred = model(phi_z, X, sample=True)

    sigma = mx.exp(model.log_sigma)
    nll = 0.5 * mx.log(2.0 * mx.pi) + model.log_sigma + 0.5 * mx.square((Y.squeeze() - pred) / sigma)
    nll = mx.mean(nll)

    kl = model.total_kl()
    kl_scaled = kl / Y.shape[0]

    return nll + kl_scaled
