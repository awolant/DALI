# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST example.

This script trains a simple Convolutional Neural Net on the MNIST dataset.
The data is loaded using tensorflow_datasets.

"""

from absl import app
from absl import flags
from absl import logging

from flax import nn
from flax import optim

import jax
from jax import random
from jax import dlpack

import jax.numpy as jnp
import numpy as onp

import tensorflow as tf

import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import os


IMAGE_SIZE = 28
NUM_CLASSES = 10


data_path = os.path.join(
    os.environ['DALI_EXTRA_PATH'], 'db/MNIST/training/')

FLAGS = flags.FLAGS
flags.DEFINE_float(
    'learning_rate', default=0.1,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=128,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=5,
    help=('Number of training epochs.'))

flags.DEFINE_boolean(
    'dali_on_cpu', default=False,
    help=('If defined run DALI pipeline on CPU'))


class MnistPipeline(Pipeline):
    def __init__(self, batch_size, num_threads=4, path=data_path, device='gpu', device_id=0, shard_id=0, num_shards=1, seed=0):
        super(MnistPipeline, self).__init__(
            batch_size, num_threads, device_id, seed)
        self.device = device
        self.reader = ops.Caffe2Reader(
            path=path, random_shuffle=True, shard_id=shard_id, num_shards=num_shards)
        self.decode = ops.ImageDecoder(
            device='mixed' if device is 'gpu' else 'cpu',
            output_type=types.GRAY)
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            output_dtype=types.FLOAT,
            image_type=types.GRAY,
            mean=[0.],
            std=[255.],
            output_layout="HWC")

    def define_graph(self):
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)
        if self.device is 'gpu':
            labels = labels.gpu()
        images = self.cmn(images)

        return images, labels


def dataset_options():
    options = tf.data.Options()
    try:
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.autotune = False   
    except:
        print('Could not set TF Dataset Options')

    return options


class CNN(nn.Module):
    """A simple CNN model."""

    def apply(self, x):
        x = nn.Conv(x, features=32, kernel_size=(3, 3))
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(x, features=64, kernel_size=(3, 3))
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(x, features=256)
        x = nn.relu(x)
        x = nn.Dense(x, features=10)
        x = nn.log_softmax(x)
        return x


def create_model(key):
    _, initial_params = CNN.init_by_shape(key, [((1, IMAGE_SIZE, IMAGE_SIZE, 1), jnp.float32)])
    model = nn.Model(CNN, initial_params)
    return model


def create_optimizer(model, learning_rate, beta):
    optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
    optimizer = optimizer_def.create(model)
    return optimizer


def onehot(labels, num_classes=10):
    x = (labels[..., None] == jnp.arange(num_classes)[None])
    return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy}
    return metrics


@jax.jit
def train_step(optimizer, batch):
    """Train for a single step."""
    def loss_fn(model):
        logits = model(batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = compute_metrics(logits, batch['label'])
    return optimizer, metrics


def train_epoch(optimizer, train_iterator, batch_size, epoch, steps_per_epoch):
    """Train for a single epoch."""

    batch_metrics = []
    for it in range(steps_per_epoch):
        batch = next(train_iterator)
        optimizer, metrics = train_step(
            optimizer, { 'image': batch[0], 'label': batch[1]})
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: onp.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}

    logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                 epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100)

    return optimizer, epoch_metrics_np


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    x = x._numpy() 
    return x

  return jax.tree_map(_prepare, xs)
  

def train(_):
    """Train MNIST to completion."""
    rng = random.PRNGKey(0)

    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs
    steps_per_epoch = 50000 // batch_size

    shapes = (
        (FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1),
        (FLAGS.batch_size))
    dtypes = (tf.float32, tf.int32)

    pipeline = MnistPipeline(batch_size, device='cpu')
    train_dataset = dali_tf.DALIDataset(
        pipeline=pipeline,
        batch_size=FLAGS.batch_size,
        output_shapes=shapes,
        output_dtypes=dtypes,
        num_threads=4,
        device_id=0)

    train_dataset = map(prepare_tf_data, train_dataset)
    train_iterator = iter(train_dataset)

    model = create_model(rng)
    optimizer = create_optimizer(model, FLAGS.learning_rate, FLAGS.momentum)

    input_rng = onp.random.RandomState(0)
    for epoch in range(1, num_epochs + 1):
        optimizer, _ = train_epoch(
            optimizer, train_iterator, batch_size, epoch, steps_per_epoch)


if __name__ == '__main__':
    app.run(train)
