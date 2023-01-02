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


@partial(jax.jit, static_argnames=['m', 'f', 'hidden_dim', 'scale'])
def collapse(params, m, f, hidden_dim, scale):
    new_params = {}
    first5x5 = collapse_lb([params['sesr/first5x5/conv2_d'],
                            params['sesr/first5x5/conv2_d_1']], 5, hidden_dim, 1, f)
    new_params['sesr/first5x5'] = wrap_no_bias(first5x5)
    for i in range(m):
        collapsed = collapse_lb([params[f'sesr/lin_{i}/conv2_d'],
                                 params[f'sesr/lin_{i}/conv2_d_1']], 3, hidden_dim, f, f)
        residual = collapse_res(collapsed)
        final = collapsed + residual
        new_params[f'sesr/conv_{i}'] = wrap_no_bias(final)
        new_params[f'sesr/prelu_{i}'] = params[f'sesr/lin_{i}/prelu']

    last5x5 = collapse_lb([params['sesr/last5x5/conv2_d'],
                           params['sesr/last5x5/conv2_d_1']], 5, hidden_dim, f, scale ** 2)
    new_params['sesr/last5x5'] = wrap_no_bias(last5x5)
    return new_params


@partial(jax.jit, static_argnames=['k', 'hidden_dim', 'n_in', 'n_out'])
def collapse_lb(w, k, hidden_dim: int, n_in: int, n_out: int):
    delta = jnp.eye(n_in)
    delta = jnp.expand_dims(jnp.expand_dims(delta, 1), 1)
    pad = k // 2
    delta = jnp.pad(delta, pad_width=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

    new_w = {'net/conv2_d': w[0], 'net/conv2_d_1': w[1]}
    lb = hk.without_apply_rng(hk.transform(lambda x: LinearBlock(kernel=k, output_dim=n_out, hidden_dim=hidden_dim)(x)))
    _ = lb.init(jax.random.PRNGKey(seed=0), delta)
    x = lb.apply(new_w, delta)

    w_c = jnp.transpose(jnp.flip(x, (1, 2)), [1, 2, 0, 3])
    return w_c


@jax.jit
def collapse_res(w_c):
    shape = w_c.shape
    outc, k = shape[3], shape[0]
    w_r = jnp.zeros(shape)
    idx = k // 2
    for i in range(outc):
        w_r = w_r.at[idx, idx, i, i].set(1)
    return w_r
