import haiku as hk
import jax.numpy as jnp
import jax

from models.linear_block import LinearBlock, ResidualLinearBlock


def collapse(params, m, f, hidden_dim):
    params = list(params.items())

    first5x5 = collapse_lb([params[0], params[1]], 5, hidden_dim, 1, f)
    for i in range(m):
        pass
    return params


def collapse_lb(w, k, hidden_dim: int, n_in: int, n_out: int):
    delta = jnp.eye(n_in)
    delta = jnp.expand_dims(jnp.expand_dims(delta, 1), 1)
    pad = k // 2
    delta = jnp.pad(delta, pad_width=[[0, 0], [pad, pad], [pad, pad], [0, 0]])
    # delta.shape = 1, 2k-1, 2k-1, 1

    w[0] = ('net/conv2_d', w[0][1])
    w[1] = ('net/conv2_d_1', w[1][1])
    lb = hk.without_apply_rng(hk.transform(lambda x: LinearBlock(kernel=k, output_dim=n_out, hidden_dim=hidden_dim)(x)))
    _ = lb.init(jax.random.PRNGKey(seed=0), delta)
    x = lb.apply(dict(w), delta)

    w_c = jnp.transpose(jnp.flip(x, (1, 2)), [1, 2, 0, 3])
    return w_c


def collapse_res(w_c):
    shape = w_c.shape
    outc, k = shape[3], shape[0]
    w_r = jnp.zeros(shape)
    idx = k // 2
    for i in range(outc):
        w_r[idx, idx, i, i] = 1
    return w_r
