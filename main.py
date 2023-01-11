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
    best_params: hk.Params
    best_psnr: float
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
    wandb.init(project="SESR-Jax", entity="sesr-jax", config=wandb_config)

    network = models.model.Model(network=FLAGS.model, should_collapse=FLAGS.collapse)
    optimiser = optax.amsgrad(5e-4)
    rng = jax.random.PRNGKey(seed=FLAGS.seed)

    @jax.jit
    def mae(x, y):
        z = jnp.mean(jnp.abs(x - y))
        return z

    @jax.jit
    def psnr(x, y):
        mse = jnp.mean((x - y) ** 2)
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
        return TrainingState(params, state.best_params, state.best_psnr, opt_state)

    def eval(eval_dataset):
        batch_mean_abs_errors = []
        batch_psnr_vals = []
        divergence_vals = []
        for batch in eval_dataset:
            batch_upscaled = network.apply(state.avg_params, batch.lr)
            batch_mean_abs_errors.append(mae(batch_upscaled, batch.hr))
            batch_psnr_vals.append(psnr(batch_upscaled, batch.hr))
        mean_abs_error = jnp.mean(jnp.array(batch_mean_abs_errors))
        psnr_val = jnp.mean(jnp.array(batch_psnr_vals))
        divergence = jnp.mean(jnp.array(divergence_vals))

        wandb.log({"epoch": iteration // iterations_per_epoch,
                   "iteration": iteration % iterations_per_epoch,
                   "mae (optimised)": float(mean_abs_error),
                   "psnr": float(psnr_val),
                   "div": float(divergence)
                   })
        logging.info({"epoch": iteration // iterations_per_epoch,
                      "iteration": iteration % iterations_per_epoch,
                      "mae (optimised)": f"{mean_abs_error:.5f}",
                      "psnr": f"{psnr_val:.3f}",
                      })
        if psnr_val > state.best_psnr:
            return TrainingState(state.params, state.params, psnr_val.item(), state.opt_state)
        return state

    # Make datasets.
    train_dataset, eval_dataset = get_dataset(dataset_name='div2k',
                                              super_res_factor=FLAGS.scale,
                                              batch_size=FLAGS.batch_size,
                                              num_crops_per_image=FLAGS.num_crops_per_image,
                                              epochs=FLAGS.epochs)
    logging.info("Created datasets.")

    # Initialise network and optimiser; note we draw an input to get shapes.
    initial_params = network.init(rng, list(train_dataset.take(1).as_numpy_iterator())[0].lr)
    initial_opt_state = optimiser.init(initial_params)
    state = TrainingState(initial_params, initial_params, 0, initial_opt_state)

    # Convert datasets to numpy for compatability with Jax
    train_dataset = tfds.as_numpy(train_dataset)
    eval_dataset = tfds.as_numpy(eval_dataset)

    iterations_per_epoch = math.ceil(NUM_TRAINING_IMAGES * FLAGS.num_crops_per_image / FLAGS.batch_size)
    iteration = 0

    logging.info("Starting training loop.")
    for batch in train_dataset:
        if iteration % iterations_per_epoch == 0:
            # Evaluate performance at the start of each epoch
            state = eval(eval_dataset)
        state = SGD(state, batch)
        iteration += 1

    logging.info(f"Training loop completed. Best PSNR = {state.best_psnr}")
    jnp.savez(f'params_{FLAGS.model}_{FLAGS.epochs}.npz', state.best_params)


if __name__ == "__main__":
    app.run(main)
