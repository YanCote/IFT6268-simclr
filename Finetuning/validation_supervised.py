import os, sys
if os.path.abspath(".") not in sys.path:
    sys.path.append(os.path.abspath("."))
import time
import dataloaders.chest_xray as chest_xray
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from sklearn.metrics import roc_auc_score
import tensorflow_hub as hub
from Finetuning.package import *
import argparse
import yaml
import pickle
import mlflow
import scipy
print(tf.__version__)
import simclr_master.resnet as resnet
tf1.disable_eager_execution()


def evaluation(yml_config, module_path=None, FLAGS=None):
    tf1.keras.backend.clear_session()
    sess = tf1.Session()
    data_path = FLAGS.xray_path
    test_dataset, tfds_info = chest_xray.XRayDataSet(data_path, config=None, train=False)
    num_images = tfds_info['num_eval_examples']
    num_images = np.floor(yml_config['finetuning']['eval_data_ratio'] * tfds_info['num_eval_examples'])
    assert yml_config['finetuning']['eval_data_ratio']
    num_classes = tfds_info['num_classes']
    batch_size = yml_config['inference']['batch']

    n_iter = int(num_images / batch_size)

    def _preprocess(x):
        x['image'] = preprocess_image(
            x['image'], 224, 224, is_training=False, color_distort=False)
        return x

    x_ds = test_dataset \
        .take(num_images) \
        .map(_preprocess, deterministic=False) \
        .batch(batch_size)\
    

    x_iter = tf1.data.make_one_shot_iterator(x_ds)
    x_init = x_iter.make_initializer(x_ds)
    x = x_iter.get_next()
    resnet.BATCH_NORM_DECAY = FLAGS.batch_norm_decay
    model = resnet.resnet_v1(resnet_depth=50,width_multiplier=1,cifar_stem=224 <= 32)


    key = model(x['image'], False)
    
    # Attach a trainable linear layer to adapt for the new task.
    with tf1.variable_scope('head_supervised_new', reuse=tf1.AUTO_REUSE):
        logits_t = tf1.layers.dense(inputs=key, units=14)
    
    cross_entropy = weighted_cel(labels=x['label'], logits=logits_t, bound = 3.0)
    loss_t = tf1.reduce_mean(tf1.reduce_sum(cross_entropy, axis=1))

    sess = tf1.Session()
    Saver = tf1.train.Saver() # Default saves all variables
    load_saver = module_path
    Saver.restore(sess, load_saver)

    with sess.as_default():
        all_labels = []
        all_logits = []
        val_tot_loss = 0
        for step in range(n_iter): 
            _loss, logits, labels = sess.run(fetches=(loss_t, logits_t, x['label']))
            logits_sig = scipy.special.expit(logits)
            all_logits.extend(logits_sig)
            all_labels.extend(labels)
            val_tot_loss += _loss

            try:
                auc_cum = roc_auc_score(np.array(all_labels),np.array(all_logits))
            except:
                auc_cum = None

            print(f" [Iter: {step}/{n_iter}] Total Loss: {val_tot_loss} Loss: {np.float32(_loss)}  AUC Cumulative: {auc_cum}")


        val_tot_loss_mean = val_tot_loss / n_iter
        try:
            epoch_auc = roc_auc_score(np.array(all_labels),np.array(all_logits), average=None)
            epoch_auc_mean = epoch_auc.mean()
            aucs = dict(zip(chest_xray.XR_LABELS.keys(),epoch_auc ))
            auc_scores = {'T-AUC ' + str(key): val for key, val in aucs.items()}

        except:
            epoch_auc= None
            epoch_auc_mean= None
        
        if yml_config['mlflow']:
            mlflow.log_metric('Total Test Loss',val_tot_loss)
            mlflow.log_metric('Avg Test Loss',val_tot_loss_mean )

            if epoch_auc is not None:
                mlflow.log_metrics(auc_scores)
                mlflow.log_metric('Avg Test AUC', epoch_auc_mean)


        print(f"Validation Done! Model: {yml_config['finetuning']['pretrained_model']}, Total Loss: {val_tot_loss}, Mean Loss: {val_tot_loss_mean},"
                f" Validation AUC: {epoch_auc_mean} AOC/Class {epoch_auc},")
    



    
