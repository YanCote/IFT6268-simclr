
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pdb
import typing
import random
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import image_ops

def load_img(path, image_size=(224, 224), num_channels=3, interpolation='bilinear'):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

def BuildDataSet(
    img_data_path: str,
    df:pd.DataFrame, 
    split_ids: [int],
    config: typing.Dict[typing.AnyStr, typing.Any] = None,
    seed: int = 1337,
    image_size: (int, int) = (224, 224),
    num_channels: int = 3, 
):
    # TODO: get config info

    def _dataset(id, img_idx, labels):
        # you have acces to dataframe here
        if img_idx is not None and img_idx != "":

            #print(img_idx)
            #pdb.set_trace()
            image_path = os.path.join(img_data_path, img_idx.decode("utf-8") )
            image_data = load_img(image_path, image_size, num_channels)

            # TODO: onehot encodings
            one_hot_labels = tf.constant([1.0, 0.0])
            yield (image_data, one_hot_labels)


    def wrap_generator(id, img_idx, labels):
        return tf.data.Dataset.from_generator(_dataset, args=[id, img_idx, labels], output_types=(tf.float64, tf.float64))

    

    # make a list of image paths to use
    patien_ids = train_df[("Patient ID")].values.tolist()
    index_imgs = train_df[("Image Index")].values.tolist()
    labels = train_df[("Finding Labels")].values.tolist()

    # Create an interleaved dataset so it's faster. Each dataset is responsible to load it's own compressed image file.
    files = tf.data.Dataset.from_tensor_slices( (patien_ids, index_imgs, labels) )
    dataset = files.interleave(wrap_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset
    

class XRayDataSet(tf.data.Dataset):
    def __new__(
        cls,
        img_data_path: typing.AnyStr,
        data_frame_path: typing.AnyStr,
        config: typing.Dict[typing.AnyStr, typing.Any] = None,
        train: bool = True,
        scratch_dir: str = None,
        seed: int = 1337,
        split: float =  0.70,
    ):
    """
    Make sure to use same random seed for training and validation datasets so they respect the data split. 
    """
        df = pd.read_csv(data_frame_path)

        # Look at dataframe and split data
        if train:
            max_id = df["Patient ID"].max()
            possible_ids = range(1, max_id + 1)
            random.seed(seed)
            split_ids = random.sample(possible_ids, int(max_id * split))
            dataframe = df[df["Patient ID"].isin(split_ids)]
        else:
            max_id = df["Patient ID"].max()
            possible_ids = range(1, max_id + 1)
            random.seed(seed) # Same seed as train, so it's the same split!
            train_samples = random.sample(possible_ids, int(max_id * split))
            split_ids = np.setdiff1d(range(1, max_id + 1), train_samples, assume_unique=True).tolist()
            dataframe = df[df["Patient ID"].isin(split_ids)]

        return BuildDataSet(img_data_path, dataframe, id_samples, config, seed)


if __name__ == "__main__":
    import PIL
    import PIL.Image
    import matplotlib.pyplot as plt

    use_cache = False
    data_frame_path = "H:/data/chest-xray/Data_Entry_2017.csv"
    img_data_path = "H:/data/chest-xray/images-224"
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
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.title("Test")
        plt.axis("off")
        plt.show()