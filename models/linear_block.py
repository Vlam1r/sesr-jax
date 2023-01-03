import haiku as hk
import jax.numpy as jnp
import jax


class LinearBlock(hk.Module):
    def __init__(self,
                 kernel: int,
                 hidden_dim: int,
                 output_dim: int,
                 name: str = 'net', ):
        super().__init__(name=name)
        self.kernel = kernel
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def __call__(self,
                 inputs: jnp.ndarray):
        conv1 = hk.Conv2D(output_channels=self.hidden_dim, kernel_shape=self.kernel, padding='VALID')
        conv2 = hk.Conv2D(output_channels=self.output_dim, kernel_shape=1, padding='VALID')

        return conv2(conv1(inputs))


class ResidualLinearBlock(LinearBlock):
    def __call__(self,
                 inputs: jnp.ndarray):
        act = jax.nn.relu
        out = super().__call__(inputs)

        return act(inputs + out)
