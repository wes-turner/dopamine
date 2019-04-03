# Asynchronous train runner.

## Overview.

The asynchronous train runner allows to train and evaluate a RL agent with an interface similar to the simple runner. On top that, the asynchronous trainer allows to:
- run experience gathering for training in multiple threads simultaneously and asynchronously
- decouple training from experience gathering so they occur simultaneously instead of sequentially.

This is particularly useful for slow environment where a `step` takes time to return a result compared to the time necessary to run a training step.
One use-case for such a trainer is to handle environments based on a RESTful API, where the `step` computation is performed by a remote server.

## Implementation details.

### Synchronization between training steps and experience gathering.

Even in an asynchronous setting, it is important that training steps and experience gathering remain simultaneous, with the same speed for experience gathering and training. This ensures that:
results are independent from different hardware set-up
training occurs through experience generation homogeneously, in the same way as for simple training.

This is achieved by adding training steps to run as experience generation goes. A training step will only be run if an additional experience step has completed.
At the same time, experience generations will be blocked if training is delayed.

### Local variables.

To handle experience generation in multiple threads simultaneously, the runner will manage variables local to each thread (including an environment object per thread).
These variables are specific to each thread id.
