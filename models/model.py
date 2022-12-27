import jax.numpy as jnp
import haiku as hk

import models.sesr as sesr


def collapse(params):
    return params

class Model:

    def expanded_fn(self, images: jnp.ndarray) -> jnp.ndarray:
        net = self.expanded()
        return net(images)

    def collapsed_fn(self, images: jnp.ndarray) -> jnp.ndarray:
        net = self.collapsed()
        return net(images)

    def __init__(self, network: str):
        assert network == 'M11'
        self.expanded = sesr.SESR_M11
        self.collapsed = sesr.SESR_M11
        self.exp_transformed = hk.without_apply_rng(hk.transform(self.expanded_fn))
        self.col_transformed = hk.without_apply_rng(hk.transform(self.collapsed_fn))

    def init(self, *args):
        return self.exp_transformed.init(*args)

    def apply(self, params, images):
        return self.col_transformed.apply(collapse(params), images)
