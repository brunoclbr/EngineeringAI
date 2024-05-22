import numpy as np
"""
- API's: Sequential Model, Functional API, Model Subclassing
- Visualizing models
- Callbacks
    - Monitoring model while training (Tensorboard)
- Reimplement fit() from scratch --> run forward pass inside gradient tape to get
    a loss value of the current batch, the retrieve the gradients w.r.t the models weights, then update them
    - then evaluate predictions with some metric
    --> leverage fit() with custom training loop
    --> class CustomModel(keras.Model):
            def train_step(seld, data):
                inputs, targets = data
                predictions = self(inputs) --> self is the keras.Model

- gen learning, self-supervised learning, reinforcement learning
    - no explicit targets  - targets are obtained from the inputs   - learning driven by rewards

- run your code eagerly if you want to debug. Add the @tf. ... decorator to run your
    code as a computaton graph (quicker)    
    
"""