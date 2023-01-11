import jax
import jax.numpy as jnp
import haiku as hk
import functools
import models.model


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
        # Don't prune prelu layers
        if 'prelu' not in k:
            weight_dict[k] = jnp.array(params[k]['w'])
            old_mask_weight_dict[k] = jnp.array(mask[k]['w'])

    num_remaining_weights = jnp.sum(jnp.array([jnp.sum(v) for v in old_mask_weight_dict.values()]))
    num_to_prune = jnp.ceil(pruning_fraction * num_remaining_weights).astype(int)

    weight_vector = jnp.concatenate([v[old_mask_weight_dict[k] == 1] for k, v in weight_dict.items()])
    threshold = jnp.sort(jnp.abs(weight_vector))[num_to_prune]

    updated_mask_dict = {k: jnp.where(jnp.abs(v) > threshold, old_mask_weight_dict[k], jnp.zeros_like(v))
                         for k, v in weight_dict.items()}

    num_after = jnp.sum(jnp.array([jnp.sum(v) for v in updated_mask_dict.values()]))
    print(f"Num Before: {num_remaining_weights} Num After: {num_after} Percentage Remaining: {num_after/num_remaining_weights}")

    for k in mask.keys():
        if 'prelu' not in k:
            mask[k]['w'] = updated_mask_dict[k]

    return mask


if __name__ == "__main__":
    m3_network = models.model.Model(network="M3", should_collapse=True)
    m3_params = jnp.load("params_M3_300.npz", allow_pickle=True)
    m3_params = m3_params['arr_0'].item(0)
    collapsed_params = jnp.load("parameters.npz", allow_pickle=True)
    collapsed_params = collapsed_params['arr_0'].item(0)
    full_mask = get_full_mask(collapsed_params)
    updated_mask = update_mask(collapsed_params, full_mask, 0.2)
    mask_applied = apply_mask(collapsed_params, updated_mask)
    print(mask_applied)

    # train_dataset, _ = data_preparation.get_dataset(dataset_name='div2k', batch_size=1)
    # train_dataset = iter(tfds.as_numpy(train_dataset))
    # first_image = next(train_dataset)
    # print(first_image)
    # print(first_image.lr)
    # np.save("low_res", first_image.lr)
    # print(hk.Params(m3_params['arr_0']))
    # test(m3_params['arr_0'])

    # print(m3_params['sesr/a'])
    # for k, v in m3_params.items():
    #     print(k['sesr/a'])
