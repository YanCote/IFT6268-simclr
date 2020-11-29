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
tf1.disable_eager_execution()


def evaluation(yml_config, args):
   
    if yml_config['mlflow']:
        #fname = yml_config['inference']['hyper_params_path']
        #with open(fname,'rb') as f:
            #params = pickle.load(f)
        mlflow.set_tracking_uri(yml_config['mlflow_path'])
        mlflow.set_experiment('validation')
        mlflow.start_run()
        mlflow.log_params(yml_config['finetuning'])

    hub_path = os.path.abspath(yml_config['inference']['hub_path'])
    module = hub.Module(hub_path, trainable=False)
    sess = tf1.Session()
    
    #TO DO  a verifier quon prend le test data setls
    if args.xray_path == '':
        data_path = yml_config['dataset']['chest_xray']
    else:
        data_path = args.xray_path
    test_dataset, tfds_info = chest_xray.XRayDataSet(data_path,train_ratio=yml_config['finetuning']['train_data_ratio'], config=None, train=False)
    num_images = tfds_info['num_eval_examples']
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

    key = module(x['image'], as_dict=True)
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=x['label'], logits=key['default'],  pos_weight=yml_config['finetuning']['pos_weight_loss'])
    loss = tf1.reduce_mean(tf1.reduce_sum(cross_entropy, axis=1))

    # ------------------------------------
    #from pathlib import Path
    #hub_path = str(Path(yml_config['finetuning']['pretrained_hub_path']).resolve())
    #module = hub.Module(hub_path, trainable=False)
    #key = module(inputs=x['image'], as_dict=True)
    #with tf1.variable_scope('head_supervised_new', reuse=tf1.AUTO_REUSE):
    #    logits_t = tf1.layers.dense(inputs=key['default'], units=num_classes)
    #    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=x['label'], logits=key['default'], pos_weight=10)
    #    loss_t = tf1.reduce_mean(tf1.reduce_sum(cross_entropy, axis=1))
#
    #variables_to_train = tf1.trainable_variables()
    #optimizer = LARSOptimizer
    #            0.2,
    #            momentum=0.1,
    #            weight_decay=0.0,
    #            exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])
    #train_op = optimizer.minimize(
    #    logits_t, global_step=tf1.train.get_or_create_global_step(),
    #    var_list=variables_to_train)
#
    #sess = tf1.Session()
    #Saver = tf1.train.Saver() # Default saves all variables
    #Saver.restore(sess, "H:/AI_Projects/outputs/runs/SimCLR/finetune/2020-11-22-22-18-51/session_24.ckpt")

    
    # ------------------------------------

    
    with sess.as_default():
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        all_labels = []
        all_logits = []
        val_tot_loss = 0
        for step in range(n_iter): 
            _loss, logits, labels = sess.run(fetches=(loss, key['default'], x['label']))
            logits_sig = scipy.special.expit(logits)
            all_logits.extend(logits_sig)
            all_labels.extend(labels)
            val_tot_loss += _loss

            try:
                auc_cum = roc_auc_score(np.array(all_labels),np.array(all_logits))
            except:
                auc_cum = None


            if yml_config['finetuning']['verbose_train_loop']:
                print(f" [Iter: {step}/{n_iter}] Total Loss: {val_tot_loss} Loss: {np.float32(_loss)}  AUC Cumulative: {auc_cum}")


        val_tot_loss_mean = val_tot_loss / n_iter
        try:
            epoch_auc = roc_auc_score(np.array(all_labels),np.array(all_logits), average=None)
            epoch_auc_mean = epoch_auc.mean()
            aucs = dict(zip(chest_xray.XR_LABELS.keys(),epoch_auc ))
            auc_scores = {'AUC ' + str(key): val for key, val in aucs.items()}

        except:
            epoch_auc= None
            epoch_auc_mean= None
        
        if yml_config['mlflow']:
            mlflow.log_metric('Total Test Loss',val_tot_loss)
            mlflow.log_metric('Avg Test Loss',val_tot_loss_mean )

            if epoch_auc is not None:
                mlflow.log_metrics(auc_scores)
                mlflow.log_metric('Avg AUC', epoch_auc_mean)


        print(f"Validation Done! Model: {yml_config['finetuning']['pretrained_model']}, Total Loss: {val_tot_loss}, Mean Loss: {val_tot_loss_mean},"
                f" Train AUC: {epoch_auc_mean} AOC/Class {epoch_auc},")
    

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

    evaluation(yml_config, args)


    
