import logging
from typing import NamedTuple
from absl import app, flags
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
import math
import models.model
import wandb
from data_preparation import get_dataset, Batch
import tensorflow as tf

tf.config.run_functions_eagerly(False)

flags.DEFINE_integer('seed', 42, 'Random seed to set')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_integer('num_crops_per_image', 64, 'Number of random crops to take from each image')
flags.DEFINE_integer('scale', 2, 'Scaling factor')
flags.DEFINE_string('model', 'M11', 'Model to train')
flags.DEFINE_boolean('collapse', True, 'Use collapsed model in forward pass')
FLAGS = flags.FLAGS
NUM_TRAINING_IMAGES = 800


class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState


def main(unused_args):
    wandb_config = dict(
        seed=FLAGS.seed,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        num_crops_per_image=FLAGS.num_crops_per_image,
        scale=FLAGS.scale,
        model=FLAGS.model,
        collapse=FLAGS.collapse
    )
    wandb.init(project="SESR-Jax", config=wandb_config)

    network = models.model.Model(network=FLAGS.model, should_collapse=FLAGS.collapse)
    optimiser = optax.amsgrad(1e-4)
    rng = jax.random.PRNGKey(seed=FLAGS.seed)

    @jax.jit
    def mae(x, y):
        z = jnp.mean(jnp.abs(x - y))
        return z

    # @jax.jit
    def psnr(x, y):
        mse = jnp.mean((x - y) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * jnp.log10(1 / jnp.sqrt(mse))

    @jax.jit
    def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        """Mean absolute error loss."""
        upscaled = network.apply(params, batch.lr)
        return mae(batch.hr, upscaled)

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
    def eval(eval_batch: Batch):
        upscaled = network.apply(state.avg_params, eval_batch.lr)
        mean_abs_error = mae(upscaled, eval_batch.hr)
        psnr_val = psnr(upscaled, eval_batch.hr)
        divergence = network.divergence(state.params, eval_batch.lr)
        wandb.log({"epoch": iteration // iterations_per_epoch,
                   "iteration": iteration,
                   "mae (optimised)": float(mean_abs_error),
                   "psnr": float(psnr_val),
                   "div": float(divergence)
                   })
        logging.info({"epoch": iteration // iterations_per_epoch,
                      "iteration": iteration,
                      "mae (optimised)": f"{mean_abs_error:.3f}",
                      "psnr": f"{psnr_val:.3f}",
                      "div": f"{divergence}"
                      })

    # Make datasets.
    rng, new_rng = jax.random.split(rng)
    train_dataset, eval_dataset = get_dataset(dataset_name='div2k',
                                              super_res_factor=FLAGS.scale,
                                              batch_size=FLAGS.batch_size,
                                              num_crops_per_image=FLAGS.num_crops_per_image,
                                              epochs=FLAGS.epochs)
    eval_dataset = list(eval_dataset.take(1).as_numpy_iterator())[0]
    logging.info("Created datasets.")

    # Initialise network and optimiser; note we draw an input to get shapes.
    initial_params = network.init(rng, list(train_dataset.take(1).as_numpy_iterator())[0].lr)
    initial_opt_state = optimiser.init(initial_params)
    state = TrainingState(initial_params, initial_params, initial_opt_state)

    # Convert dataset to numpy for compatability with Jax
    train_dataset = tfds.as_numpy(train_dataset)

    iterations_per_epoch = math.ceil(NUM_TRAINING_IMAGES * FLAGS.num_crops_per_image / FLAGS.batch_size)
    iteration = 0

    logging.info("Starting training loop.")
    for batch in train_dataset:
        if iteration % 10 == 0:
            eval(eval_dataset)
        state = SGD(state, batch)
        iteration += 1

    logging.info("Training loop completed.")
    jnp.savez('params.npz', state.params)


if __name__ == "__main__":
    app.run(main)
