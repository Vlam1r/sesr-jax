import haiku as hk
import jax
import jax.numpy as jnp
import prelu


class SmallResidualBlock(hk.Module):
    def __init__(self,
               f: int,
               name: str = 'net',):
        super().__init__(name=name)
        self.f = f

    def __call__(self, inputs):
        conv = hk.Conv2D(output_channels=self.f, kernel_shape=[3, 3])
        return prelu.PRELU(inputs + conv(inputs))

class SESR(hk.Module):
    def __init__(self,
                 m: int,
                 f: int,
                 scale: int,
                 name: str = 'net',
                 ):
        super().__init__(name=name)
        self.m = m
        self.f = f
        self.scale = scale

    def __call__(self, inputs):
        first5x5 = hk.Conv2D(output_channels=self.f, kernel_shape=(5,5))
        residual_blocks = [SmallResidualBlock(self.f)] * self.m
        ...



        pass
