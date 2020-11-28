rm -rf output
out_dir="/Users/yancote/mila/IFT6268-simclr/output"
pretrain_dir="output"
dt=$(date '+%d-%m-%Y-%H-%M-%S');
out_dir=$pretrain_dir/${dt}

echo 'Time Signature: ${dt}'
echo "Saving Monolytic File Archive in : ${out_dir}/run_${dt}.txt"
if 
python -u  ./simclr_master/run.py --data_dir "NIH/" \
--train_batch_size 2 \
--optimizer adam \
--model_dir $out_dir \
--use_multi_gpus \
--checkpoint_path $out_dir \
--momentum 0.9 \
--learning_rate 0.1 \
--use_blur \
--temperature 0.5 \
--proj_out_dim 128 \
--train_data_split 0.00005 \
--train_epochs 10 --checkpoint_epochs 50 \
--weight_decay 0.0 --warmup_epochs 0 \
--color_jitter_strength 0.5 | tee run_${dt}.txt;
# --color_jitter_strength 0.5 > run_${dt}.txt 2>&1;

then
echo 'Time Signature: $dt'
echo "Saving Monolytic File Archive in : ${out_dir}/run_${dt}.txt"
cp run_${dt}.txt "${out_dir}/run_${dt}.txt"

cd $pretrain_dir
tar -zcvf ${dt}.tar.gz ${dt}
fi
echo 'PreTraining Completed !!! '