# Asynchronous train runner.

## Overview.

With the asynchronous trainer a user can train and evaluate a RL agent with an interface similar to the simple trainer. On top that, the asynchronous trainer allows to:
- run experience gathering for training in multiple threads simultaneously and asynchronously
- decouple training from experience gathering so they occur simultaneously instead of sequentially.

This is particularly useful for slow environment where a `step` takes much more time to return a result than the time necessary to run a training step.
One use-case for such a trainer is to handle environments based on a RESTful API, where the `step` computation is performed by a remote server.

## Implementation details.

### Synchronization between training steps and experience gathering.

Even in an asynchronous setting, it is important that training steps and experience gathering remain simultaneous, with the same speed (same speed in average with some flexibility for small variations to maintain uniformity) for experience gathering and training.
This ensures that:
- results are independent from different hardware set-up
- training occurs through experience generation uniformly, in the same way as for simple training.

An example of undesired behavior is if experience gathering runs 2x faster than training. In that case, once experience gathering completes there are still half the training steps to complete. The last generated samples will have been computed using a policy half way through training, and the second half of the training steps will not be used to generate any new sample. This means there is no feedback loop for the second half of training.

Simultaneous training is achieved by adding training step tasks to a queue as experience generation goes. A training task is enqueued at each experience generation step, and dequeued in the thread running training steps. Therefore a training step will only be run if an additional experience step has completed (if there is a training step task to dequeue).
This queue has a size limit for the number of training tasks cached, which ensures experience generations is blocked if training is delayed (and the queue is full).
The size limit of the queue is set to the number of experience generation thread to handle the case where all thread complete at the same time, without blocking.

### Local variables.

To handle experience generation in multiple threads simultaneously, the runner manages variables local to each thread (including an environment object per thread).
These variables are specific to each thread id.
