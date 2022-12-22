import logging
import pickle
from typing import Iterator, NamedTuple

from absl import app, flags
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import model.sesr
from data_preparation import get_dataset, Batch

import tensorflow as tf
tf.config.run_functions_eagerly(False)

flags.DEFINE_integer('seed', 42, 'Random seed to set')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train')
flags.DEFINE_integer('scale', 2, 'Scaling factor')
FLAGS = flags.FLAGS


class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState


def net_fn(images: jnp.ndarray) -> jnp.ndarray:
    """Standard LeNet-300-100 MLP network."""
    x = images.astype(jnp.float32) / 255.
    fn = model.sesr.SESR_M11()
    return fn(x)

# @jax.jit
def PSNR(x, y):
    mse = jnp.mean((x - y) ** 2)
    max_pixel = 255.0
    psnr = 20 * jnp.log10(max_pixel / jnp.sqrt(mse))
    return jnp.nan_to_num(psnr, nan=100)



def main(unused_args):

    network = hk.without_apply_rng(hk.transform(net_fn))
    optimiser = optax.adam(1e-3)
    rng = jax.random.PRNGKey(seed=FLAGS.seed)

    @jax.jit
    def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        """Mean absolute error loss."""
        upscaled = network.apply(params, batch.lr)
        #mae = jnp.mean(jnp.abs(batch.hr - upscaled))
        mse = jnp.mean((batch.hr - upscaled) ** 2)
        return mse

    @jax.jit
    def SGD(state: TrainingState, batch: Batch) -> TrainingState:
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(loss)(state.params, batch)
        updates, opt_state = optimiser.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        # Compute avg_params, the exponential moving average of the "live" params.
        # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
        avg_params = optax.incremental_update(
            params, state.avg_params, step_size=0.001)
        return TrainingState(params, avg_params, opt_state)

    # @jax.jit
    def eval():
        mse = loss(state.avg_params, next(eval_dataset))
        logging.info({"step": step, "mse": f"{mse:.3f}"})



    # Make datasets.
    rng, new_rng = jax.random.split(rng)
    train_dataset, eval_dataset = get_dataset(dataset_name='div2k',
                                              super_res_factor=FLAGS.scale)
    logging.info("Created datasets.")

    # Initialise network and optimiser; note we draw an input to get shapes.
    initial_params = network.init(rng, next(train_dataset).lr)
    initial_opt_state = optimiser.init(initial_params)
    state = TrainingState(initial_params, initial_params, initial_opt_state)

    logging.info("Starting training loop.")
    # Training & evaluation loop.
    for step in range(FLAGS.epochs):
        if step % 10 == 0:
            eval()

        # Do SGD on a batch of training examples.
        state = SGD(state, next(train_dataset))

    logging.info("Training loop completed.")


    with open('model.pkl', 'wb') as f:
        pickle.dump({'params': state.params, 'opt_state': state.opt_state}, f)


if __name__ == "__main__":
    app.run(main)
