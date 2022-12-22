import haiku as hk
import jax
import jax.numpy as jnp


class PRELU(hk.Module):
    """
    Parametrised ReLU implementation for Haiku
    """
    def __init__(self,
                 name: str = 'prelu',):
        super().__init__(name=name)

    def __call__(self, inputs):
        alpha = hk.get_parameter("alpha", shape=[1], init=jnp.zeros)
        output = jax.nn.leaky_relu(x=inputs, negative_slope=alpha)
        return output
