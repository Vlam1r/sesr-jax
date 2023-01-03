from typing import Dict, Any

import jax
import jax.numpy as jnp
import haiku as hk

import models.sesr_collapsed as sesr_c
from models.collapser import collapse, make_collapsers, collapse_res, wrap_no_bias


def get_sesr_args(name: str):
    return {
        'M3': {'m': 3, 'f': 16, 'hidden_dim': 256, 'scale': 2},
        'M5': {'m': 5, 'f': 16, 'hidden_dim': 256, 'scale': 2},
        'M11': {'m': 11, 'f': 16, 'hidden_dim': 64, 'scale': 2},
        'XL': {'m': 11, 'f': 32, 'hidden_dim': 64, 'scale': 2},
        }.get(name)


class Model:

    def __init__(self, network: str):
        self.kwargs = get_sesr_args(network)
        self.net = hk.without_apply_rng(hk.transform(lambda x: sesr_c.SESR_Collapsed(**self.kwargs)(x)))
        self.collapser = make_collapsers(**self.kwargs)
        self.collapser_map = {c[0]: (c[1], c[2]) for c in self.collapser}
        self.collapsed_params = None

    def init(self, rng: jax.random.PRNGKeyArray, image: jnp.ndarray) -> Dict[str, Any]:
        """Initialise each collapser"""
        out = {}
        for i in self.collapser:
            k, c, delta = i
            rng, new_rng = jax.random.split(rng)
            v = c.init(new_rng, delta)
            out[k] = v
        self.update(out)
        # _ = self.net.init(rng, image)
        return out

    def forward(self, params, images):
        return self.net.apply(collapse(params, self.collapser), images)

