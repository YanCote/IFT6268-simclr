from __future__ import absolute_import, division, print_function, unicode_literals

# -*- coding: utf-8 -*-
"""finetuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/google-research/simclr/blob/master/colabs/finetuning.ipynb

# Copyright 2020 Google LLC.
"""

# @title Licensed under the Apache License, Version 2.0 (the "License");
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""<a href="https://colab.research.google.com/github/google-research/simclr/blob/master/colabs/finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

This colab demonstrates how to load pretrained/finetuned SimCLR models from hub modules for fine-tuning

The checkpoints are accessible in the following Google Cloud Storage folders.

* Pretrained SimCLRv2 models with a linear classifier: [gs://simclr-checkpoints/simclrv2/pretrained](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/pretrained)
* Fine-tuned SimCLRv2 models on 1% of labels: [gs://simclr-checkpoints/simclrv2/finetuned_1pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/finetuned_1pct)
* Fine-tuned SimCLRv2 models on 10% of labels: [gs://simclr-checkpoints/simclrv2/finetuned_10pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/finetuned_10pct)
* Fine-tuned SimCLRv2 models on 100% of labels: [gs://simclr-checkpoints/simclrv2/finetuned_100pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/finetuned_100pct)
* Supervised models with the same architectures: [gs://simclr-checkpoints/simclrv2/pretrained](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/pretrained)

Use the corresponding checkpoint / hub-module paths for accessing the model. For example, to use a pre-trained model (with a linear classifier) with ResNet-152 (2x+SK), set the path to `gs://simclr-checkpoints/simclrv2/pretrained/r152_2x_sk1`.
"""
import os, sys
if os.path.abspath(".") not in sys.path:
    sys.path.append(os.path.abspath("."))

import dataloaders.chest_xray as chest_xray
import utils.model_ckpt as model_ckpt
import warnings
warnings.filterwarnings("ignore")
import argparse
import yaml
from tensorflow.python.client import device_lib
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
#import tensorflow_datasets as tfds
import tensorflow_hub as hub
import re
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import mlflow
import pickle
from pathlib import Path
import time
import scipy
from Finetuning.package import *
print(tf.__version__)
tf1.disable_eager_execution()
from shutil import rmtree
from functools import partial
import simclr_master.resnet as resnet
from utils.create_folder import create_folder

def test_weighted_cel():
    with tf1.Session():
        b = 64
        c = 14
        logits = tf.random.uniform([b, c], minval=-1, maxval=1)
        labels = tf.cast(tf.random.uniform([b, c], minval=-1, maxval=1) > 0, tf.float32)
        loss = weighted_cel(labels=labels, logits=logits)
        pass

def show_one_image(im):
    plt.imshow(im)
    plt.title("Test")
    plt.axis("off")
    plt.show()

def train(args, yml_config):
    with strategy.scope():

        # @title Load tensorflow datasets: we use tensorflow flower dataset as an examplegit
        batch_size = yml_config['finetuning']['batch']
        buffer_size = yml_config['finetuning']['buffer_size']

        # @title Load tensorflow datasets: we use tensorflow flower dataset as an example
        dataset_name = yml_config['data_src']



        if dataset_name == 'tf_flowers':
            tfds_dataset, tfds_info = tfds.load(
                dataset_name, split='train', with_info=True)
            num_images = tfds_info.splits['train'].num_examples
            num_classes = tfds_info.features['label'].num_classes

            x = tfds_dataset.map(_preprocess).batch(batch_size)
            x = tf1.data.make_one_shot_iterator(x).get_next()

        elif dataset_name == 'chest_xray':
            if args.xray_path == '':
                data_path = yml_config['dataset']['chest_xray']
            else:
                data_path = args.xray_path
            train_dataset, tfds_info = chest_xray.XRayDataSet(data_path, config=None, train=True)
            num_images = np.floor(yml_config['finetuning']['train_data_ratio'] *tfds_info['num_examples'])
            num_classes = tfds_info['num_classes']
            
        print(f"Training: {num_images} images...")



        def _preprocess(x):
            x['image'] = preprocess_image(
                x['image'], 224, 224, is_training=False, color_distort=False)
            return x
            
        x_ds = train_dataset \
            .take(num_images) \
            .map(_preprocess, deterministic=False) \
            .shuffle(buffer_size)\
            .batch(yml_config['finetuning']['batch'])


        x_iter = tf1.data.make_one_shot_iterator(x_ds)
        x_init = x_iter.make_initializer(x_ds)
        x = x_iter.get_next()

        print(f"{type(x)} {type(x['image'])} {x['image']} {x['label']}")
        # @title Load module and construct the computation graph
        learning_rate = yml_config['finetuning']['learning_rate']
        momentum = yml_config['finetuning']['momentum']
        weight_decay = yml_config['finetuning']['weight_decay']
        epoch_save_step = yml_config['finetuning']['epoch_save_step']
        load_saver = yml_config['finetuning'].get('load_ckpt')

        # Load the base network and set it to non-trainable (for speedup fine-tuning)
        hub_path = str(Path(yml_config['finetuning']['pretrained_hub_path']).resolve())
        module = hub.Module(hub_path, trainable=yml_config['finetuning']['train_resnet'])
        
        if  yml_config['finetuning']['pretrained_model'] == 'ChestXRay':
            key = module(inputs=x['image'], signature="projection-head-1", as_dict=True)
        else:
            key = module(inputs=x['image'], as_dict=True)


        # Attach a trainable linear layer to adapt for the new task.
        if dataset_name == 'tf_flowers':
            with tf1.variable_scope('head_supervised_new', reuse=tf1.AUTO_REUSE):
                logits_t = tf1.layers.dense(inputs=key['default'], units=num_classes, name='proj_head')
            loss_t = tf1.reduce_mean(input_tensor=tf1.nn.softmax_cross_entropy_with_logits(
                labels=tf1.one_hot(x['label'], num_classes), logits=logits_t))
        elif dataset_name == 'chest_xray':
            with tf1.variable_scope('head_supervised_new', reuse=tf1.AUTO_REUSE):
                #logits_t = tf1.layers.dense(inputs=key['final_avg_pool'], units=num_classes)
                logits_t = tf1.layers.dense(inputs=key['default'], units=num_classes)
                cross_entropy = weighted_cel(labels=x['label'], logits=logits_t, bound = 3.0)
                #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=x['label'], logits=logits_t, pos_weight=yml_config['finetuning']['pos_weight_loss'])
                #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x['label'], logits=logits_t)
                loss_t = tf1.reduce_mean(tf1.reduce_sum(cross_entropy, axis=1))


        # Setup optimizer and training op.
        if yml_config['finetuning']['optimizer'] == 'adam':
            optimizer = tf1.train.AdamOptimizer(learning_rate)
        elif yml_config['finetuning']['optimizer'] == 'lars':
            optimizer = LARSOptimizer(
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])
        else:
            raise RuntimeError("Optimizer not supported")


        variables_to_train = tf1.trainable_variables()
        train_op = optimizer.minimize(
            loss_t, global_step=tf1.train.get_or_create_global_step(),
            var_list=variables_to_train)

        print('Variables to train:', variables_to_train)
        key  # The accessible tensor in the return dictionary

        # Add ops to save and restore all the variables.
        sess = tf1.Session()
        Saver = tf1.train.Saver() # Default saves all variables
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        directory = Path(yml_config['checkpoint_dir'])/current_time

        is_time_to_save_session = partial(model_ckpt.save_session, epoch_save_step, Saver, output=directory)
        if load_saver is not None:
            Saver.restore(sess, load_saver)
        else:
            sess.run(tf1.global_variables_initializer())

        # @title We fine-tune the new *linear layer* for just a few iterations.
        epochs = yml_config['finetuning']['epochs']


        # ===============Tensor board section ===============
        # with tf.name_scope('performance'):
        # tf_labels = tf1.placeholder(tf.int32, shape=[batch_size,num_classes], name='accuracy')
        tf_tot_acc_all_ph = tf1.placeholder(tf.float32, shape=None, name='accuracy_all_labels_ph')
        tf_tot_acc_all_summary = tf1.summary.scalar('accuracy_all_labels', tf_tot_acc_all_ph)
        tf_tot_acc_per_class_ph = tf1.placeholder(tf.float32, shape=None, name='accuracy_per_class_ph')
        tf_tot_acc_per_class_summary = tf1.summary.scalar('accuracy_per_class', tf_tot_acc_per_class_ph)
        tf_tot_acc_class_avg_ph = tf1.placeholder(tf.float32, shape=None, name='accuracy_per_class_averaged_ph')
        tf_tot_acc_class_avg_summary = tf1.summary.scalar('accuracy_per_class_averaged', tf_tot_acc_class_avg_ph)
        tf_train_tot_loss_ph = tf1.placeholder(tf.float32, shape=None, name='train_tot_loss')
        tf_train_tot_loss_summary = tf1.summary.scalar('train_tot_loss', tf_train_tot_loss_ph)
        tf_tot_auc_ph = tf1.placeholder(tf.float32, shape=None, name='auc_ph')
        tf_tot_auc_ph_summary = tf1.summary.scalar('auc', tf_tot_auc_ph)

        performance_summaries = tf1.summary.merge(
            [tf_tot_acc_all_summary, tf_tot_acc_class_avg_summary, tf_train_tot_loss_summary, tf_tot_auc_ph_summary])

        hyper_param = []
        for item in yml_config['finetuning']:
            hyper_param.append(tf1.summary.text(str(item), tf.constant(str(yml_config['finetuning'][item])),'HyperParam'))

        summ_writer = tf1.summary.FileWriter(Path(yml_config['tensorboard_path']) / current_time, sess.graph)
        tf.summary.record_if(yml_config['tensorboard'])
        # Limit the precision of floats...
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        with sess.as_default() as scope:
            if yml_config['mlflow']:
                fname = str(directory / f'params.pickle')
                create_folder(directory)
                with open(fname,'wb') as f:
                    pickle.dump(yml_config['finetuning'], f)
                mlflow.set_tracking_uri(yml_config['mlflow_path'])
                mlflow.set_experiment('fine_tuning')
                mlflow.start_run()
                mlflow.log_artifact(fname)
                mlflow.log_param('TB_Timestamp', current_time)
                mlflow.log_param('Train or Test', 'Train')
                mlflow.log_params(yml_config['finetuning'])
            writer = tf1.summary.FileWriter('./log', sess.graph)
            for index,summary_op in enumerate(hyper_param):
                text = sess.run(summary_op)
                summ_writer.add_summary(text, index)


            n_iter = int(num_images / batch_size)
            print(f"Batch:{batch_size}, n_iter:{n_iter} ")

            # =============== Main Loop (epoch) - START ===============
            for it in range(epochs):
                start_time_epoch = time.time()
                # Init dataset iterator
                sess.run(x_init)
                # Accuracy all = All class must be Correct
                # Accuracy per class = Score for each class
                # Accuracy class average: the average of the accuracy per class
                tot_acc_all = 0.0
                tot_acc_per_class = 0.0
                tot_acc_class_avg = 0.0
                train_tot_loss = 0.0
                epoch_acc_all = 0.0
                epoch_acc_per_class = 0.0
                epoch_acc_class_avg = 0.0
                #show_one_image(x['image'][0].eval())

                # =============== Main Loop (iteration) - START ===============
                all_labels = []
                all_logits = []
                for step in range(n_iter):

                    start_time_iter = time.time()
                    _, loss, image, logits, labels = sess.run(fetches=(train_op, loss_t, x['image'], logits_t, x['label']))
                    # tf_labels = tf.convert_to_tensor(labels)
                    train_tot_loss += loss
                    all_labels.extend(labels)
                    if dataset_name == 'tf_flowers':
                        pred = logits.argmax(-1)
                        correct = np.sum(pred == labels)
                        acc_per_class = np.array([correct / float(batch_size)])
                    elif dataset_name == 'chest_xray':
                        # # New compute
                        logits_sig = scipy.special.expit(logits)
                        all_logits.extend(logits_sig)
                        pred = (logits_sig > 0.5).astype(np.float32)
                        acc_all = np.mean(np.min(np.equal(pred, labels).astype(np.float32), axis=1))
                        acc_per_class = np.mean(np.equal(pred, labels).astype(np.float32), axis=0)
                        acc_class_avg = np.mean(acc_per_class)
                        tot_acc_all += acc_all
                        tot_acc_per_class += acc_per_class
                        tot_acc_class_avg += acc_class_avg

                    #The function roc_auc_score can result in a error (ValueError: Only one class present in y_true.
                    # ROC AUC score is not defined in that) . The error occurred when each label has only one class
                    # in the batch. For example, if all the samples in the batch has hernia +1, the error will occurred.I
                    try:
                        auc_cum = roc_auc_score(np.array(all_labels),np.array(all_logits))
                    except:
                        auc_cum = None

                    current_time_iter = time.time()
                    elapsed_time_iter = current_time_iter - start_time_iter

                    if yml_config['finetuning']['verbose_train_loop']:
                        print(f"[Epoch {it + 1}/{epochs} Iter: {step}/{n_iter}] Model: {yml_config['finetuning']['pretrained_model']}, Total Loss: {train_tot_loss} Loss: {np.float32(loss)} Batch Acc: {np.float32(acc_all)} "
                              f"Acc Avg(class): {np.float32(acc_class_avg)}, AUC Cumulative: {auc_cum}")
                        print(f"Finished iteration:{step} in: " + str(int(elapsed_time_iter)) + f" sec logits min,max: {np.min(logits)},{np.max(logits)}")
                    # =============== Main Loop (iteration) - END ===============
                    if np.isnan(np.sum(logits)):
                        print(f"Loss has exploded: Nan")
                        it = epochs
                        break
                epoch_acc_all = (tot_acc_all/n_iter)
                epoch_acc_per_class = (tot_acc_per_class / n_iter)
                epoch_acc_class_avg = (tot_acc_class_avg / n_iter)


                try:
                    epoch_auc = roc_auc_score(np.array(all_labels),np.array(all_logits), average=None)
                    epoch_auc_mean = epoch_auc.mean()
                    aucs = dict(zip(chest_xray.XR_LABELS.keys(),epoch_auc ))
                    auc_scores = {'AUC ' + str(key): val for key, val in aucs.items()}

                except:
                    epoch_auc= None
                    epoch_auc_mean= None

                print(f"[Epoch {it + 1}/{epochs} Model: {yml_config['finetuning']['pretrained_model']}, Loss: {train_tot_loss} Train Acc: {np.float32(epoch_acc_all)}, Train Acc Avg(class) {np.float32(epoch_acc_class_avg)}"
                      f" Train AUC: {epoch_auc_mean} AOC/Class {epoch_auc},")
                
                # Is it time to save the session?
                is_time_to_save_session(it, sess)

                current_time_epoch = time.time()
                elapsed_time_iter = current_time_epoch - start_time_epoch
                print(f"Finished EPOCH:{it + 1} in: " + str(int(elapsed_time_iter)) + " sec")
                # print(psutil.virtual_memory())

                # ===================== Write Tensorboard summary ===============================
                # Execute the summaries defined above

                summ = sess.run(performance_summaries, feed_dict={tf_tot_acc_all_ph: epoch_acc_all,
                                                                  tf_tot_acc_class_avg_ph: epoch_acc_class_avg,
                                                                  tf_train_tot_loss_ph: train_tot_loss,
                                                                  tf_tot_auc_ph: epoch_auc_mean})


                # Write the obtained summaries to the file, so it can be displayed in the TensorBoard
                summ_writer.add_summary(summ, it)

                # =============== Main Loop (epoch) - END ===============

            print(f"Training Done")


            # This MLFLOW code is now saving training metrics. When the validation accuracy will be completed,
            # we should save instead the validation/test metrics.
            # The saving will occured only at the end of the finetuning
            if yml_config['mlflow']:
                mlflow.log_metric('Total Train Accuracy',epoch_acc_all)
                mlflow.log_metric('Total Train Accuracy per class', np.mean(epoch_acc_per_class))
                mlflow.log_metric('Total Train Loss',train_tot_loss)

                if epoch_auc is not None:
                    mlflow.log_metrics(auc_scores)

            #rmtree(str(Path.cwd() / yml_config['finetuning']['finetuned_cp']))
            #ckpt_pt = Saver.save(sess=sess,save_path=str(Path.cwd() / yml_config['finetuning']['finetuned_cp'] / 'pt'), global_step=step)
            #print(f"Final Chekpoint Saved in {yml_config['finetuning']['finetuned_cp']}")


def build_hub_module(yml_config, num_classes, hub_id_name, checkpoint_path, save_path):
    """Create TF-Hub module."""

    tags_and_args = [
        # The default graph is built with batch_norm, dropout etc. in inference
        # mode. This graph version is good for inference, not training.
        ([], {'is_training': False}),
        # A separate "train" graph builds batch_norm, dropout etc. in training
        # mode.
        (['train'], {'is_training': True}),
    ]

    def module_fn(is_training):
        endpoints = {}
        inputs = tf1.placeholder(
            tf1.float32, [None, None, None, 3])

        # Load the base network and set it to non-trainable (for speedup fine-tuning)
        hub_path = str(Path(yml_config['finetuning']['pretrained_hub_path']).resolve())
        module = hub.Module(hub_path, trainable=is_training)

        if  yml_config['finetuning']['pretrained_model'] == 'ChestXRay':
            key = module(inputs=inputs, signature="projection-head-1", as_dict=True)
        else:
            key = module(inputs=inputs, as_dict=True)


        # Attach a trainable linear layer to adapt for the new task.
        with tf1.variable_scope('head_supervised_new', reuse=tf1.AUTO_REUSE):
            logits_t = tf1.layers.dense(inputs=key['default'], units=num_classes)
            endpoints['head_classification'] = logits_t
        
        hub.add_signature(inputs=dict(images=inputs),
                          outputs=dict(endpoints, default=logits_t))

    spec = hub.create_module_spec(module_fn, tags_and_args)
    hub_export_dir = os.path.join(save_path, 'hub')
    checkpoint_export_dir = os.path.join(hub_export_dir, str(hub_id_name))
    create_folder(checkpoint_export_dir)
    spec.export(
        checkpoint_export_dir,
        checkpoint_path=checkpoint_path,
        name_transform_fn=None)

    return hub_export_dir

def create_module_from_checkpoints(yml_config):
    hub_id_name = 'chest_xray'
    num_classes = 14
    checkpoint_path = yml_config['finetuning'].get('load_ckpt')
    save_path = os.path.abspath("./saved_modules")
    p = build_hub_module(yml_config, num_classes, dataset_name, checkpoint_path, save_path)

    print("Module hub saved at {0} with id name {1}".format(p, hub_id_name))

def main(args, yml_config, save_hub=False):
    if not save_hub:
        train(args, yml_config)
    else:
        create_module_from_checkpoints(yml_config)


if __name__ == "__main__":
    
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Finetuning on SimClrv2')
    parser.add_argument('--config', '-c', default='config.yml', required=False,
                    help='yml configuration file')
    parser.add_argument('--xray_path', default='', required=False,
                        help='yml configuration file')
    parser.add_argument('--save_hub', action='store_true', default=False,
                        help='yml configuration file')
    args = parser.parse_args()

    # Yaml configuration files
    try:
        with open(args.config) as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        raise RuntimeError(f"Configuration file {args.config} do not exist")

    # tf1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

    if tf.config.list_physical_devices('gpu'):
        strategy = tf.distribute.CentralStorageStrategy()
    else:  # use default strategy
        strategy = tf.distribute.get_strategy()


    # Profiler
    # tf.profiler.experimental.server.start(6009)
    # tf.profiler.experimental.client.trace('grpc://127.0.0.1:6009',
    #                                       'gs://local_dir', 2000)

    main(args, yml_config, args.save_hub)

    