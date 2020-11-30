import os, sys
from absl import flags
import tensorflow as tf
import tensorflow_hub as hub
from absl import app
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

if os.path.abspath(".") not in sys.path:
    sys.path.append(os.path.abspath("."))
import dataloaders.chest_xray as chest_xray
from simclr_master.data import build_chest_xray_fn
import eval_ssl.test_tools as test_tools

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_enum(
    'train_mode', 'eval', ['pretrain', 'finetune', 'eval'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')


def main(ar):
    data_path = "H:/data/chest-xray"

    chest_xray_dataset = build_chest_xray_fn(True, data_path, None, True)({'batch_size': 10})
    #chest_xray_dataset, info = chest_xray.XRayDataSet(data_path, config=None, train=False)
    for x in chest_xray_dataset.take(10):
        fig=plt.figure()
        plt.imshow(x[0][0])
        plt.axis("off")
        plt.show()

    sys.exit()

if __name__ == "__main__":
    app.run(main)



