import jax
import jax.numpy as jnp
import haiku as hk


import models.sesr as sesr
import models.sesr_collapsed as sesr_c
from models.collapser import collapse
import pruning


def get_sesr_args(name: str):
    return {
        'M3': {'m': 3, 'f': 16, 'hidden_dim': 256, 'scale': 2},
        'M5': {'m': 5, 'f': 16, 'hidden_dim': 256, 'scale': 2},
        'M11': {'m': 11, 'f': 16, 'hidden_dim': 64, 'scale': 2},
        'XL': {'m': 11, 'f': 32, 'hidden_dim': 64, 'scale': 2},
        }.get(name)


class Model:

    def expanded_fn(self, images: jnp.ndarray, **kwargs) -> jnp.ndarray:
        net = self.expanded(**kwargs)
        return net(images)

    def collapsed_fn(self, images: jnp.ndarray, **kwargs) -> jnp.ndarray:
        net = self.collapsed(**kwargs)
        return net(images)

    def __init__(self, network: str, should_collapse: bool):
        self.kwargs = get_sesr_args(network)
        self.expanded = sesr.SESR
        self.collapsed = sesr_c.SESR_Collapsed
        self.should_collapse = should_collapse
        self.exp_transformed = hk.without_apply_rng(hk.transform(self.expanded_fn))
        self.col_transformed = hk.without_apply_rng(hk.transform(self.collapsed_fn))

    def init(self, *args):
        return self.exp_transformed.init(*args, **self.kwargs)

    def apply(self, params, mask, images):
        if self.should_collapse:
            collapsed_params = collapse(params, **self.kwargs)
            collapsed_params = pruning.apply_mask(collapsed_params, mask)
            return self.col_transformed.apply(collapsed_params, images, **self.kwargs)
        else:
            return self.exp_transformed.apply(params, images, **self.kwargs)