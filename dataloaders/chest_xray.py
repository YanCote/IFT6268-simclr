
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pdb
import typing
import random
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import image_ops

XR_LABELS = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Consolidation': 2,
    'Edema': 3,
    'Effusion': 4,
    'Emphysema': 5,
    'Fibrosis': 6,
    'Hernia': 7,
    'Infiltration': 8,
    'Mass': 9,
    'No Finding': 10,
    'Nodule': 11,
    'Pleural_Thickening': 12,
    'Pneumonia' : 13,
    'Pneumothorax': 14,
}

xray_n_class = len(XR_LABELS.keys())
verbose = 0
img_dtype = tf.float32
# img_dtype = tf.int32

def load_img(path, one_hot_labels):
    image_size=(224, 224)
    num_channels=3
    interpolation='bilinear'

    # Image
    img = io_ops.read_file(path)
    img = tf.compat.v1.image.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = tf.compat.v1.image.resize(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))

    return {'image': img, 'label': one_hot_labels}

def BuildDataSet(
    img_data_path: str,
    df:pd.DataFrame, 
    config: typing.Dict[typing.AnyStr, typing.Any] = None,
    seed: int = 1337,
    image_size: (int, int) = (224, 224),
    num_channels: int = 3, 
):
    # TODO: get config info

    # make a list of image paths to use
    index_imgs = df[("Image Index")].values.tolist()
    labels = df[("Finding Labels")].values.tolist()
    for i in range(len(index_imgs)):
        index_imgs[i] = os.path.join(img_data_path, index_imgs[i])

    # Make onehot labels
    one_hot_labels = xray_n_class*[0]
    for i in range(len(labels)):
        for key in XR_LABELS.keys():
            one_hot_labels[XR_LABELS[key]] = 1 if key in labels[i] else 0
        labels[i] = tf.cast(one_hot_labels, dtype=tf.float32)

    # Create an interleaved dataset so it's faster. Each dataset is responsible to load it's own compressed image file.
    dataset = tf.data.Dataset.from_tensor_slices( (index_imgs, labels) )
    #dataset = files.interleave(wrap_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(load_img)

    # TODO change num classes
    # < YC use dict's len 29/10/2020>
    return dataset, {"num_examples": df.shape[0], "num_classes": xray_n_class}
    

class XRayDataSet(tf.data.Dataset):
    def __new__(
        cls,
        data_path: typing.AnyStr,
        config: typing.Dict[typing.AnyStr, typing.Any] = None,
        train: bool = True,
        seed: int = 1337,
        split: float =  0.10,
    ):
        """
        Make sure to use same random seed for training and validation datasets so they respect the data split. 
        """
        df = pd.read_csv(os.path.join(data_path, "Data_Entry_2017.csv"))

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

        img_data_path = os.path.join(data_path, "images-224")
        return BuildDataSet(img_data_path, dataframe, config, seed)


if __name__ == "__main__":
    import PIL
    import PIL.Image
    import matplotlib.pyplot as plt

    use_cache = False
    # data_frame_path = "./NIH/Data_Entry_2017.csv"
    data_path = "../NIH"
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
        train_ds , tfds_info = XRayDataSet(data_path, config=config,train=True) \
            # .prefetch(tf.data.experimental.AUTOTUNE) \
            # .shuffle(buffer_size)\
            # .batch(batch_size)

    # [x['image'].shape for x in train_ds.take(20)]
            
    for data in train_ds.take(20):
        plt.imshow(data['image'].numpy().astype("uint8"))
        plt.title("Test")
        plt.axis("off")
        plt.show()
        print(data['label'])