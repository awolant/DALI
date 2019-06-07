# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import unittest
import os
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class TestDaliTfPluginLoadOk(unittest.TestCase):
    def test_import_dali_tf_ok(self):
        import nvidia.dali.plugin.tf as dali_tf
        assert True

class TestDaliTfPluginLoadFail(unittest.TestCase):
    def test_import_dali_tf_load_fail(self):
        with self.assertRaises(Exception):
            import nvidia.dali.plugin.tf as dali_tf

test_data_root = os.environ['DALI_EXTRA_PATH']
file_root = os.path.join(test_data_root, 'db', 'coco', 'images')
annotations_file = os.path.join(test_data_root, 'db', 'coco', 'instances.json')

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def show_images(image_batch, nb_images):
    columns = 4
    rows = (nb_images + 1) // (columns)
    fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(nb_images):
        plt.subplot(gs[j])
        plt.axis("off")
        img = image_batch[0][j].transpose((1,2,0)) + 128
        plt.imshow(img.astype('uint8'))

class COCOPipeline(Pipeline):
    def __init__(self, batch_size):
        super(COCOPipeline, self).__init__(batch_size, 4, 0, seed = 15)
        self.input = ops.COCOReader(
            file_root = file_root,
            annotations_file = annotations_file,
            shard_id = 0, 
            num_shards = 1, 
            ratio=False, 
            save_img_ids=True)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        self.resize = ops.Resize(
            device = "cpu",
            image_type = types.RGB,
            interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(
            device = "cpu",
            output_dtype = types.FLOAT,
            crop = (224, 224),
            image_type = types.RGB,
            mean = [128., 128., 128.],
            std = [1., 1., 1.])
        self.res_uniform = ops.Uniform(range = (256.,480.))
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.cast = ops.Cast(
            device = "cpu",
            dtype = types.INT32)

    def define_graph(self):
        inputs, bboxes, labels, im_ids = self.input()
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter = self.res_uniform())
        output = self.cmn(
            images, 
            crop_pos_x = self.uniform(),
            crop_pos_y = self.uniform())
        output = self.cast(output)
        return (output, bboxes, labels, im_ids)

# class TestDaliTfRead(unittest.TestCase):
class TestDaliTfRead():
    def test_run(self):
        import tensorflow as tf
        import nvidia.dali.plugin.tf as dali_tf

        batch_size = 8

        pipe = COCOPipeline(batch_size)

        shapes = [(batch_size, 3, 224, 224), (), (), ()]
        dtypes = [tf.int32, tf.float32, tf.int32, tf.int32]
        sparse = [False, True, True, True]

        daliop = dali_tf.DALIIterator()
        

        images = []
        bboxes = []
        labels = []
        image_ids = []
        with tf.device('/cpu'):
            # image, bbox, label, id = daliop(
            # next_element = daliop(
            #     pipeline = pipe,
            #     shapes = shapes,
            #     dtypes = dtypes, 
                # sparse = sparse)

            daliset = dali_tf.DALIDataset(
                pipeline = pipe,
                batch_size = batch_size,
                sparse = sparse,
                shapes = shapes,
                dtypes = dtypes,
                devices = [0])
            iterator = daliset.make_one_shot_iterator()
            next_element = iterator.get_next()

            # images.append(image)
            # bboxes.append(bbox)
            # labels.append(label)
            # image_ids.append(id)


            with tf.Session() as sess:

                # The actual run with our dali_tf tensors
                # res_cpu = sess.run([images, bboxes, labels, image_ids])
                res_cpu = sess.run(next_element)
            print(res_cpu)
            # print(res_cpu[2])
            # print(res_cpu[3])
            # print(res_cpu)

        # show_images(res_cpu[0], 8)


if __name__ == '__main__':
    # unittest.main()
    test = TestDaliTfRead()

    test.test_run()
