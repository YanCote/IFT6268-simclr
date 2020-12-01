device_type             : ['/cpu:0', '/gpu:0']
data_src                : 'chest_xray'
tensorboard             : True
mlflow_path             :  /home/yancote1/IFT6268-simclr/mlruns   #/home/gauthies/IFT6268-simclr/mlruns
mlflow                  : True
imagenet_lemma_path     : ./Finetuning/ilsvrc2012_wordnet_lemmas.txt

enum:
    optimizer: ['momentum', 'adam', 'lars']
    data_src_type: ['imagenet', 'chest_xray','tf_flowers']

dataset:
    flower_ts_data_path : "./"
    chest_xray : $SLURM_TMPDIR

finetuning:
    epochs                  : 1
    batch                   : 64
    optimizer               : 'lars'
    learning_rate           : 0.01
    momentum                : 0.9
    weight_decay            : 0.1
    buffer_size             : 256
    epoch_save_step         : 20                        # Save checkpoints every n epochs
    train_data_ratio        : 0.001                       # Portion of the Data to use
    train_resnet            : False                     # trainable flag for the resnet model
    verbose_train_loop      : False
    eval_data_ratio         : 0.001
    pretrained_build        :  '/home/yancote1/models/pretrain/30-11-2020-10-08-27'
    pretrained_model        : 'ChestXRay' # ChestXRay or 'ImageNetSSl'



inference:
    batch                    : 5
    pretrained_hub_path      : './r50_1x_sk0/'


distillation:
    batch                   : 32
    total_steps             : 10
    learning_rate           : 2.0
    momentum                : 0.9
    weight_decay            : 1e-4
    temperature             : 1
    hub_path                : './r50_1x_sk0/hub/'
