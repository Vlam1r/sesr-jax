Reimplementation of [Collapsible Linear Blocks for Super-Efficient Super Resolution](https://arxiv.org/pdf/2103.09404v4.pdf) in JAX.

Please see `planning_and_decision_making.md` for a description of how the project evolved.

There are 3 key branches:
1. `master` - Our re-implementation of the paper from scratch. This includes an updated method for collapsing blocks which takes into account biases correctly, unlike code release by the authors ([https://github.com/ARM-software/sesr/issues/14](https://github.com/ARM-software/sesr/issues/14)) 
2. `new-sesr` - An implementation of the model which more closely resembles the authors' implementation.
3. `pruning` - A branch which takes our re-implementation, and allows the model to be pruned (i.e. have certain weights set to 0), using iterative magnitude pruning as described in [The Lottery Ticket Hypothesis](https://arxiv.org/pdf/1803.03635.pdf).

We recommend using a virtual environment for the project, and installing requirements listed in `requirements.txt`.

Use the command `python main.py` to run the main training loop, and run the command `python main.py --help` to see details on which flags can be set for the training run (such as the number of epochs and model type).

The code for the model can be found in the `models` folder. The main model is defined in `model.py`, and utilises the classes defined in `sesr.py` and `sesr_collapsed.py` based on whether the `collapse` flag is set to true or false.

For pruning the model on the `pruning` branch, we utilise a binary mask, and perform an element-wise multiplication with the weights of the model on the forward pass to "prune" weights.

Some visualisation of images upsampled by the model can be found in the `upscaling_visualisation/data_visualisation.ipynb` notebook. For visualisation of upsampled images from the pruned model, which are used in the report, see [https://github.com/Vlam1r/sesr-jax/blob/pruning/data_visualisation.ipynb](https://github.com/Vlam1r/sesr-jax/blob/pruning/data_visualisation.ipynb).

