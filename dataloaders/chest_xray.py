
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
    b'Atelectasis': 0,
    b'Cardiomegaly': 1,
    b'Consolidation': 2,
    b'Edema': 3,
    b'Effusion': 4,
    b'Emphysema': 5,
    b'Fibrosis': 6,
    b'Hernia': 7,
    b'Infiltration': 8,
    b'Mass': 9,
    b'No Finding': 10,
    b'Nodule': 11,
    b'Pleural_Thickening': 12,
    b'Pneumonia' : 13,
    b'Pneumothorax': 14,
}

xray_n_class = len(XR_LABELS.keys())
verbose = 0
img_dtype = tf.float32
# img_dtype = tf.int32

def load_img(path, image_size=(224, 224), num_channels=3, interpolation='bilinear'):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    # tf.image.convert_image_dtype(
    #     img, dtype=tf.float32, saturate=False, name=None)
    print(img.dtype)
    return img.dtype

def BuildDataSet(
    img_data_path: str,
    df:pd.DataFrame, 
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
            one_hot_labels = xray_n_class*[0]
            # TODO: onehot encodings
            # < YC one hot Encoding first implementation 29/10/2020>
            for key in XR_LABELS.keys():
                one_hot_labels[XR_LABELS[key]] = 1 if key in labels else 0
            one_hot_labels = 4
            if verbose:
                print(f"{labels} {image_data.shape} {one_hot_labels}")
            yield {'image': image_data, 'label':one_hot_labels}





    def wrap_generator(id, img_idx, labels):
        return tf.data.Dataset.from_generator(_dataset, args=[id, img_idx, labels], output_types={'image': tf.float32, 'label': tf.int64},
                                              output_shapes={'image': tf.TensorShape([224, 224, 3]), 'label': tf.TensorShape([xray_n_class])})

    # make a list of image paths to use
    patient_ids = df[("Patient ID")].values.tolist()
    index_imgs = df[("Image Index")].values.tolist()
    labels = df[("Finding Labels")].values.tolist()

    # Create an interleaved dataset so it's faster. Each dataset is responsible to load it's own compressed image file.
    files = tf.data.Dataset.from_tensor_slices( (patient_ids, index_imgs, labels) )
    dataset = files.interleave(wrap_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
        split: float =  0.70,
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