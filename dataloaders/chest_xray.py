
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pdb
import typing
import random
from tensorflow.python.ops import io_ops
from sklearn.preprocessing import MultiLabelBinarizer

num_class = 14
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
    'Nodule': 10,
    'Pleural_Thickening': 11,
    'Pneumonia' : 12,
    'Pneumothorax': 13
    # 'No Finding': 14
}

xray_n_class = len(XR_LABELS.keys())
verbose = 0
img_dtype = tf.float32
# img_dtype = tf.int32

def load_img(path, one_hot_labels, img_idx):
    image_size = (224, 224)
    num_channels = 3
    interpolation = 'bilinear'

    # Image
    img = io_ops.read_file(path)
    img = tf.compat.v1.image.decode_image(
        img, channels=num_channels, expand_animations=False)
    #img = tf.compat.v1.image.resize(img, image_size, method=interpolation)
    #img.set_shape((image_size[0], image_size[1], num_channels))

    return {'image': img, 'label': one_hot_labels, 'idx': img_idx}

def PrepareData(
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
    labels_split = [labels[i].split('|') for i in range(len(df[("Finding Labels")]))]

    for i in range(len(index_imgs)):
        index_imgs[i] = os.path.join(img_data_path, index_imgs[i])


    mlb = MultiLabelBinarizer(classes = list(XR_LABELS.keys()))
    one_hot_labels = mlb.fit_transform(labels_split)

    label_cardinality = np.zeros(num_class)
    for col in range(one_hot_labels.shape[1]):
        label_cardinality[col] = np.where(one_hot_labels[:,col] == 1)[0].shape[0]
    np.count_nonzero(label_cardinality)
    if np.count_nonzero(label_cardinality) != num_class:
        print (f"At least one Class has no label instance in the data, class{np.where(label_cardinality == 0)[0].tolist()}")
        print(f"Data_cnt: {one_hot_labels.shape[0]}")
        raise RuntimeError("Not all Labels are present")
        pass
        # TODO change num classes
    # < YC use dict's len 29/10/2020>
    return (index_imgs, tf.convert_to_tensor(one_hot_labels, dtype=tf.float32), df[("Image Index")].values.tolist())
    

class XRayDataSet(tf.data.Dataset):
    """
    Wrapper class for the NIH ChestRay Dataset: https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    it is a Multi-label(14) classification

    """
    def __new__(
        cls,
        data_path: typing.AnyStr,
        config: typing.Dict[typing.AnyStr, typing.Any] = None,
        train: bool = True,
        seed: int = 1337,
        split: float = 0.90,
        return_tf_dataset: bool = True,
    ):
        """
        Make sure to use same random seed for training and validation datasets so they respect the data split. 
        """
        df = pd.read_csv(os.path.join(data_path, "Data_Entry_2017.csv"))
        # Look at dataframe and split data, generate info
        if train:
            max_id = df["Patient ID"].max()
            possible_ids = range(1, max_id + 1)
            random.seed(seed)
            split_ids = random.sample(possible_ids, int(max_id * split))
            train_df = df[df["Patient ID"].isin(split_ids)]
            
            dataframe = train_df.sample(frac=1).reset_index(drop=True) # shuffle data
            num_examples = train_df.shape[0]
            num_eval_examples = df.shape[0] - num_examples
        else:
            max_id = df["Patient ID"].max()
            possible_ids = range(1, max_id + 1)
            random.seed(seed) # Same seed as train, so it's the same split!
            train_samples = random.sample(possible_ids, int(max_id * split))
            split_ids = np.setdiff1d(range(1, max_id + 1), train_samples, assume_unique=True).tolist()
            dataframe = df[df["Patient ID"].isin(split_ids)]

            num_eval_examples = dataframe.shape[0]
            num_examples = df.shape[0] - num_eval_examples

        # Do we just want the info, or do we want a new dataset. 
        if return_tf_dataset:
            img_data_path = os.path.join(data_path, "images-224")
            data = PrepareData(img_data_path, dataframe, config, seed)
            dataset = tf.data.Dataset.from_tensor_slices( data )
            return_data = dataset.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            return_data = None

        return return_data, {"num_examples": num_examples, 
                            "num_classes": xray_n_class,
                            "num_eval_examples": num_eval_examples}


if __name__ == "__main__":
    import PIL
    import PIL.Image
    import matplotlib.pyplot as plt

    use_cache = False
    data_path = "../NIH"
    # data_path = "H:/data/chest-xray"
    cache_dir = './cache'
    config = dict()
    scratch_dir = None
    batch_size = 128
    buffer_size = 128 * 2

    if use_cache:
        train_ds = XRayDataSet(data_path) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .batch(batch_size) \
            .cache(cache_dir + "/tf_learn_cache") \
            .shuffle(buffer_size)

    else:
        train_ds , tfds_info = XRayDataSet(data_path, config=config, train=True) 

    # [x['image'].shape for x in train_ds.take(20)]
            
    for data in train_ds.take(20):
        plt.imshow(data['image'].numpy().astype("uint8"))
        plt.title("Test")
        plt.axis("off")
        plt.show()
        print(data['label'])