"""
Generate sample data for air pollution
From literature, air pollution correlates with poor brain health
"""

import mlx.core as mx

# air pollution data
mx.random.seed(42)
X = mx.random.uniform(-2.0, 2.0, (100, 1))

# brain health scores
# relationship: brain health = -1.5 * Pollution + 0.5 + random noise
noise = mx.random.normal((100, 1)) * 0.3
Y = (-1.5 * X) + 0.5 + noise
