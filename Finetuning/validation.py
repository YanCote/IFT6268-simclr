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
import mlflow
print(tf.__version__)
tf1.disable_eager_execution()


def evaluation(yml_config, args):

    if yml_config['mlflow']:
                mlflow.set_tracking_uri(yml_config['mlflow_path'])
                mlflow.set_experiment('validation')
                mlflow.start_run()
                #TO DO LOG PARAM FROM FILE
                #mlflow.log_params(yml_config['finetuning'])

    hub_path = os.path.abspath(yml_config['inference']['hub_path'])
    module = hub.Module(hub_path, trainable=False)
    sess = tf1.Session()
    
    #TO DO  a verifier quon prend le test data set
    data_path = yml_config['dataset']['chest_xray']
    test_dataset, tfds_info = chest_xray.XRayDataSet(data_path,train_ratio=yml_config['finetuning']['train_data_ratio'], config=None, train=False)
    num_images = tfds_info['num_examples']
    num_classes = tfds_info['num_classes']
    batch_size = yml_config['finetuning']['batch']
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

    
    with sess.as_default():
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        all_labels = []
        all_logits = []
        val_tot_loss = 0
        for step in range(n_iter): 
            x = x_iter.get_next()
            logits = module(x['image']).eval()
            labels = x['label'].eval()
            all_labels.extend(labels)
            all_logits.extend(logits)
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, pos_weight=yml_config['finetuning']['pos_weight_loss'])
            loss = tf1.reduce_mean(tf1.reduce_sum(cross_entropy, axis=1))
            val_tot_loss += loss

            try:
                auc_cum = roc_auc_score(np.array(all_labels),np.array(all_logits))
            except:
                auc_cum = None


            if yml_config['finetuning']['verbose_train_loop']:
                print(f" [Iter: {step}/{n_iter}] Total Loss: {val_tot_loss.eval()} Loss: {np.float32(loss.eval())}  AUC Cumulative: {auc_cum}")


        val_tot_loss_mean = val_tot_loss / n_iter
        try:
            epoch_auc = roc_auc_score(np.array(all_labels),np.array(all_logits), average=None)
            epoch_auc_mean = epoch_auc.mean()
            aucs = dict(zip(chest_xray.XR_LABELS.keys(),epoch_auc ))
            auc_scores = {'AUC ' + str(key): val for key, val in aucs.items()}

        except:
            epoch_auc= None
            epoch_auc_mean= None

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
        with open('./Finetuning/'+args.config,'r') as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        raise RuntimeError(f"Configuration file {args.config} do not exist")

    evaluation(yml_config, args)


    