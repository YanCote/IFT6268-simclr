from __future__ import absolute_import, division, print_function, unicode_literals

from dataloaders.chest_xray import XRayDataSet
import tensorflow as tf

if __name__ == "__main__":

    use_cache = False
    data_frame_path = "~/scratch/data/Data_Entry_2017.csv"
    img_data_path = "~/scratch/data/chest-xray/images-224"
    config = dict()
    scratch_dir = None
    batch_size = 128
    buffer_size = 128 * 2

    if use_cache:
        train_ds = XRayDataSet(img_data_path, data_frame_path, config=config, scratch_dir=scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) \
            .cache(cache_dir + "/tf_learn_cache") \
            .shuffle(buffer_size)

    else:
        train_ds = XRayDataSet(img_data_path, data_frame_path, config=config, scratch_dir=scratch_dir) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .shuffle(buffer_size)\
            .batch(batch_size) 
            
    for (images, labels) in train_ds:
        tf.print(images[0], output_stream=sys.stdout)
        print()
        print("Just printed image. Done!!")
        break
