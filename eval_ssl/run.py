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

flags.DEFINE_float(
    'color_jitter_strength', 0.5,
    'The strength of color jittering.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune', 'eval'],
    'The train mode controls different objectives and trainable components.')

def main(argv):
    tf.compat.v1.disable_eager_execution()

    data_path = "H:/data/chest-xray"
    hub_path = os.path.abspath('C:/Users/Game/AI/SSL/project/IFT6268-simclr/saved_modules/hub/chest_xray')#("./r50_1x_sk0/hub")
    module = hub.Module(hub_path, trainable=False)

    sess = tf.compat.v1.Session()
    
    nb_of_patients = 100
    features = {}

    chest_xray_dataset = build_chest_xray_fn(True, data_path, None, False)({'batch_size': 100}).prefetch(1)
    _, info = chest_xray.XRayDataSet(data_path, train=False)
    chest_xray_dataset_itr = tf.compat.v1.data.make_one_shot_iterator(chest_xray_dataset)
    x = chest_xray_dataset_itr.get_next()
    chest_xray_dataset_init = chest_xray_dataset_itr.make_initializer(chest_xray_dataset)
    with sess.as_default():
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(10): 
            # Keep a total of 200 patients
            if len(features) >= nb_of_patients:
                break
            x1, x2 = tf.split(x[0], 2, -1)
            feat1, feat2, idx_s = sess.run(fetches=(module(x1), module(x2), x[1].get('idx')))
            for i in range(feat1.shape[0]):
                if len(features) >= nb_of_patients:
                    break
                idx = idx_s[i].decode("utf-8").split("_")[0]
                # Only add a single example for each patient. 
                if features.get(idx) is None:
                    features[idx] = []
                    features[idx].append(feat1[i])
                    features[idx].append(feat2[i])

    # Save hardwork
    output = os.path.abspath("./eval_ssl/model_out")
    Path(output).mkdir(parents=True, exist_ok=True)
    file_name = os.path.join(output, 'outputs_{}.pickle'.format("test"))
    print("Saving outputs in: {}".format(file_name))
    with open(file_name, 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Calculate mAP
    print("Calculating mAPs...")
    mAPs, quantiles = test_tools.test_mAP_outputs(epochs=[features], with_filepath=False)
    if mAPs is not None:
        print("\nResults:")
        print("mAP: {}, var: {}, quantiles 0.2: {}, median: {}, 0.8: {}".format(
            mAPs[0][0], mAPs[0][1], quantiles[0][0], quantiles[0][1], quantiles[0][2]))
    
    test_tools.display_relevantcy_barchart(epochs=[features], with_filepath=False)
    
    sys.exit(0)

if __name__ == "__main__":
    app.run(main)