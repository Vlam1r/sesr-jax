import jax
import jax.numpy as jnp
import haiku as hk
import functools


def apply_mask(params: hk.Params, mask: hk.Params) -> hk.Params:
    """Apply mask to parameters i.e. prune weights corresponding to 0s in the mask"""
    return jax.tree_util.tree_map(lambda x, y: x * y, params, mask)


def get_ones(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.ones(x.shape)


def get_full_mask(tree: hk.Params) -> hk.Params:
    """Takes a PyTree representing the parameters of a neural network and generates a corresponding mask PyTree
        which is full of ones (i.e. unpruned)"""
    map_fn = functools.partial(get_ones)
    mask = jax.tree_util.tree_map(map_fn, tree)
    return mask


def update_mask(params: hk.Params, mask: hk.Params, pruning_fraction: float) -> hk.Params:
    """Prune the smallest (abs) 'pruning_fraction' weights, ignoring biases. Based on implementation
     https://github.com/facebookresearch/open_lth/blob/main/pruning/sparse_global.py"""
    weight_dict = {}
    old_mask_weight_dict = {}
    for k in params.keys():
        weight_dict[k] = jnp.array(params[k]['w'])
        old_mask_weight_dict[k] = jnp.array(mask[k]['w'])

    num_remaining_weights = jnp.sum(jnp.array([jnp.sum(v) for v in old_mask_weight_dict.values()]))
    num_to_prune = jnp.ceil(pruning_fraction * num_remaining_weights).astype(int)

    weight_vector = jnp.concatenate([v[old_mask_weight_dict[k] == 1] for k, v in weight_dict.items()])
    threshold = jnp.sort(jnp.abs(weight_vector))[num_to_prune]

    updated_mask_dict = {k: jnp.where(jnp.abs(v) > threshold, old_mask_weight_dict[k], jnp.zeros_like(v))
                         for k, v in weight_dict.items()}

    for k in mask.keys():
        mask[k]['w'] = updated_mask_dict[k]

    return mask
