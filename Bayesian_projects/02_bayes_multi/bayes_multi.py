## new model
import mlx.core as mx
import mlx.nn as nn
import math

# multi class bayesian mlp layers
class BayesianLinear(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()

        # initialise the mean using std initialization bounds
        bound = 1.0 / math.sqrt(input_dim)
        self.w_mu = mx.random.uniform(-bound, bound, (input_dim, output_dim))
        self.b_mu = mx.random.uniform(-bound, bound, (output_dim,))

        # initialise rho to a flat negative constant
        self.w_rho = mx.full((input_dim, output_dim), -3.0)
        self.b_rho = mx.full((output_dim), -3.0)

    def __call__(self, x):
        w_sigma = mx.log(1.0 + mx.exp(self.w_rho))
        b_sigma = mx.log(1.0 + mx.exp(self.b_rho))

        epsilon_w = mx.random.normal(self.w_mu.shape)
        epsilon_b = mx.random.normal(self.b_mu.shape)
        
        # reparameterization trick
        sampled_weight = self.w_mu + w_sigma * epsilon_w
        sampled_bias = self.b_mu + b_sigma * epsilon_b

        # Track KLD against std normal prior
        self.current_kl = self._kl(sampled_weight, self.w_mu)

        return mx.matmul(x, sampled_weight) + sampled_bias

class BayesianClassificationMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layer1 = BayesianLinear(in_dim, hidden_dim)
        self.layer2 = BayesianLinear(hidden_dim, num_classes)

    def __call__(self, x):
        x = mx.maximum(self.layer1(x), 0.0)
        return self.layer2(x)
    
    def total_kl(self):
        return self.layer1.current_kl + self.layer2.current_kl