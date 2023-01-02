import haiku as hk
import jax.numpy as jnp

import models.prelu


class SESR_Collapsed(hk.Module):
    def __init__(self,
                 m: int,
                 f: int,
                 scale: int,
                 hidden_dim: int,
                 name: str = 'sesr',
                 ):
        super().__init__(name=name)
        self.m = m
        self.f = f
        self.hidden_dim = hidden_dim
        self.scale = scale

    def __call__(self,
                 inputs: jnp.ndarray):
        first5x5 = hk.Conv2D(output_channels=self.f,
                             kernel_shape=5,
                             name='first5x5')  # First 5x5 linear block
        residual_blocks = hk.Sequential(sum([[hk.Conv2D(output_channels=self.f,
                                                   kernel_shape=3,
                                                   name=f'conv_{i}'),
                                         models.prelu.PReLU(name=f'prelu_{i}')]
                                         for i in range(self.m)], []))  # M 3x3 residual linear blocks
        last5x5 = hk.Conv2D(output_channels=self.scale ** 2,
                              kernel_shape=5,
                              name='last5x5')  # Last 5x5 linear block

        a = first5x5(inputs)
        b = a + residual_blocks(a)  # First long residual connection
        c = last5x5(b)  # Second long residual connection
        d = inputs + c
        e = self.depth_to_space(d)
        return jnp.clip(e[..., jnp.newaxis], a_min=0., a_max=1.)  # Add singleton channel and clips output

    def depth_to_space(self,
                       inputs: jnp.ndarray):
        n, w, h, c = inputs.shape
        # assert c == self.scale**2
        x = jnp.reshape(inputs, (n, w, h, self.scale, self.scale))
        x = jnp.swapaxes(x, 2, 3)
        x = jnp.reshape(x, (n, w * self.scale, h * self.scale))
        return x
