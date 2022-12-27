import haiku as hk
import jax
import jax.numpy as jnp
from models.linear_block import LinearBlock, ResidualLinearBlock


class SESR(hk.Module):
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
        first5x5 = LinearBlock(hidden_dim=self.hidden_dim,
                               output_dim=self.f,
                               kernel=5)  # First 5x5 linear block
        residual_blocks = hk.Sequential([ResidualLinearBlock(hidden_dim=self.hidden_dim,
                                                             output_dim=self.f,
                                                             kernel=3)] * self.m)  # M 3x3 residual linear blocks
        last5x5 = LinearBlock(hidden_dim=self.hidden_dim,
                              output_dim=self.scale ** 2,
                              kernel=5)  # Last 5x5 linear block

        a = first5x5(inputs)
        b = a + residual_blocks(a)  # First long residual connection
        c = last5x5(b)  # Second long residual connection
        d = inputs + c
        e = self.depth_to_space(d)
        return e.reshape(*e.shape, 1)  # Add bogus 1 channel at the end

    # @jax.jit
    def depth_to_space(self,
                       inputs: jnp.ndarray):
        n, w, h, c = inputs.shape
        # assert c == self.scale**2
        x = jnp.reshape(inputs, (n, w, h, self.scale, self.scale))
        x = jnp.swapaxes(x, 2, 3)
        x = jnp.reshape(x, (n, w * self.scale, h * self.scale))
        return x


class SESR_M3(SESR):
    def __init__(self,
                 scale: int = 2,
                 name: str = 'sesr_m3'):
        super().__init__(scale=scale, m=3, f=16, hidden_dim=256, name=name)


class SESR_M5(SESR):
    def __init__(self,
                 scale: int = 2,
                 name: str = 'sesr_m5'):
        super().__init__(scale=scale, m=5, f=16, hidden_dim=256, name=name)


class SESR_M11(SESR):
    def __init__(self,
                 scale: int = 2,
                 name: str = 'sesr_m11'):
        super().__init__(scale=scale, m=11, f=16, hidden_dim=256, name=name)  # TODO or f=32?
