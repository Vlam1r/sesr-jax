from typing import Dict, List, Tuple

import haiku as hk
import jax.numpy as jnp
import jax
from functools import partial

from models.linear_block import LinearBlock


@jax.jit
def wrap_no_bias(w):
    return {
        'w': w,
        'b': jnp.zeros(shape=(w.shape[-1]))
        }


@partial(jax.jit, static_argnames=['collapser'])
def collapse(expanded_params, collapser):
    params = list(expanded_params.items())
    # First collapse each param by feeding delta through collapser
    params = [(tup[0], collapser[idx][1].apply(tup[1], collapser[idx][2])) for idx, tup in enumerate(
        params)]
    # Also reshape
    params = [(p[0], jnp.transpose(jnp.flip(p[1], (1, 2)), [1, 2, 0, 3])) for p in params]
    # Second add residuals if needed
    for i in range(1, len(params) - 1):
        residual = collapse_res(params[i][1])
        params[i] = params[i][0], params[i][1] + residual
    # Third add empty bias
    params = [(tup[0], wrap_no_bias(tup[1])) for tup in params]

    return {tup[0]: tup[1] for tup in params}


# @jax.jit
def make_delta(n_in: int, k: int) -> jnp.ndarray:
    delta = jnp.eye(n_in)
    delta = jnp.expand_dims(jnp.expand_dims(delta, 1), 1)
    pad = k - 1
    delta = jnp.pad(delta, pad_width=[[0, 0], [pad, pad], [pad, pad], [0, 0]])
    return delta


def make_collapsers(m: int, f: int, hidden_dim: int, scale: int) -> List[Tuple[str, hk.Transformed, jnp.ndarray]]:
    out = []

    def add(n_in, n_out, k, name):
        out.append(('sesr/' + name,
                    hk.without_apply_rng(hk.transform(lambda x: LinearBlock(kernel=k, output_dim=n_out,
                                                                            hidden_dim=hidden_dim, name=name)(x))),
                    make_delta(n_in, k)
                    ))

    add(1, f, 5, 'a')
    for i in range(m):
        add(f, f, 3, f'b_{i:02d}')
    add(f, scale ** 2, 5, 'c')
    return out


@jax.jit
def collapse_res(w_c):
    shape = w_c.shape
    outc, k = shape[3], shape[0]
    w_r = jnp.zeros(shape)
    idx = k // 2
    for i in range(outc):
        w_r = w_r.at[idx, idx, i, i].set(1)
    return w_r
