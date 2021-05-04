from model import YOLOv4Model, calc_loss
import numpy as np
import tensorflow as tf

from img import read_img, draw_img
from pipeline import YOLOv4Pipeline
import utils

import math
import os
import random


SET_MEMORY_GROWTH = False


class SaveWeightsCallback(tf.keras.callbacks.Callback):

    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def on_epoch_begin(self, epoch, logs=None):
        self.model.save_weights(self.ckpt_dir + '/epoch_' + str(epoch) + '.h5')


# TODO: fix nan loss issue
def train(file_root, annotations_file, batch_size, epochs, steps_per_epoch, **kwargs):

    seed = kwargs.get("seed")
    if not seed:
        seed = int.from_bytes(os.urandom(4), "little")
    else:
        os.environ['PYTHONHASHSEED']=str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


    if SET_MEMORY_GROWTH:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dali_use_gpu = kwargs.get("dali_use_gpu")
    use_mosaic = kwargs.get("use_mosaic")
    log_dir = kwargs.get("log_dir")
    ckpt_dir = kwargs.get("ckpt_dir")
    start_weights = kwargs.get("start_weights")

    total_steps = epochs * steps_per_epoch
    initial_lr = kwargs.get("lr")
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [int(0.3 * total_steps), int(0.5 * total_steps)],
        [initial_lr, 0.1 * initial_lr, 0.01 * initial_lr]
    )

    initial_epoch = 0

    multigpu = kwargs.get("multigpu")
    strategy = tf.distribute.MirroredStrategy() if multigpu else tf.distribute.get_strategy()

    with strategy.scope():
        model = YOLOv4Model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_fn)
        )

    if start_weights:
        model.load_weights(start_weights)
        fn = start_weights.split('/')[-1]
        if fn.endswith('.h5') and fn.startswith('epoch_'):
            initial_epoch = int(fn[6 : -3])


    def dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            num_threads = input_context.num_input_pipelines
            image_size = (608, 608)

            pipeline = YOLOv4Pipeline(
                file_root, annotations_file,
                batch_size, image_size, num_threads, device_id, seed,
                dali_use_gpu=dali_use_gpu,
                is_training=True,
                use_mosaic=use_mosaic
            )
            return pipeline.dataset()

    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device = True,
        experimental_prefetch_to_device = False,
        experimental_replication_mode = tf.distribute.InputReplicationMode.PER_REPLICA)

    dataset = strategy.distribute_datasets_from_function(dataset_fn, input_options)


    callbacks = []
    if log_dir:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq='epoch'
        ))
    if ckpt_dir:
        callbacks.append(SaveWeightsCallback(ckpt_dir))


    model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    return model
