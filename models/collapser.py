import haiku as hk
import jax.numpy as jnp
import jax
from functools import partial

from models.linear_block import LinearBlock


@partial(jax.jit, static_argnames=['m'])
def collapse(params, m, **unused_kwargs):
    new_params = {}
    first5x5 = collapse_lb([params['sesr/first5x5/conv2_d'],
                            params['sesr/first5x5/conv2_d_1']])
    new_params['sesr/first5x5'] = first5x5
    for i in range(m):
        collapsed = collapse_lb([params[f'sesr/lin_{i}/conv2_d'],
                                 params[f'sesr/lin_{i}/conv2_d_1']])
        residual = collapse_res(collapsed['w'])
        final = collapsed['w'] + residual
        new_params[f'sesr/conv_{i}'] = {'w': final, 'b': collapsed['b']}
        new_params[f'sesr/prelu_{i}'] = params[f'sesr/lin_{i}/prelu']

    last5x5 = collapse_lb([params['sesr/last5x5/conv2_d'],
                           params['sesr/last5x5/conv2_d_1']])
    new_params['sesr/last5x5'] = last5x5
    return new_params


@jax.jit
def collapse_lb(w):
    w_1 = w[0]['w']
    w_2 = w[1]['w']
    b_1 = w[0]['b']
    b_2 = w[1]['b']
    k = w_1.shape[0]
    n_in = w_1.shape[2]
    hidden_dim = w_1.shape[3]
    n_out = w_2.shape[3]
    delta = jnp.eye(n_in)
    delta = jnp.expand_dims(jnp.expand_dims(delta, 1), 1)
    pad = k // 2
    delta = jnp.pad(delta, pad_width=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

    new_w = {'net/conv2_d': {'b': jnp.zeros_like(b_1), 'w': w_1},
             'net/conv2_d_1': {'b': jnp.zeros_like(b_2), 'w': w_2}}
    lb = hk.without_apply_rng(hk.transform(lambda x: LinearBlock(kernel=k, output_dim=n_out, hidden_dim=hidden_dim)(x)))
    x = lb.apply(new_w, delta)

    w_c = jnp.transpose(jnp.flip(x, (1, 2)), [1, 2, 0, 3])

    # Fix the original algorithm by also computing bias terms

    conv = hk.without_apply_rng(hk.transform(lambda x: hk.Conv2D(kernel_shape=1, output_channels=n_out, name='net')(x)))
    conv_weights = {'net': {'w': w_2, 'b': b_2}}
    b_c = conv.apply(conv_weights, b_1.reshape(1, 1, 1, -1)).flatten()
    return {'w': w_c, 'b': b_c}


@jax.jit
def collapse_res(w_c):
    shape = w_c.shape
    outc, k = shape[3], shape[0]
    w_r = jnp.zeros(shape)
    idx = k // 2
    for i in range(outc):
        w_r = w_r.at[idx, idx, i, i].set(1)
    return w_r
