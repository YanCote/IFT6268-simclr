rm -rf output
out_dir="/Users/yancote/mila/IFT6268-simclr/output"
pretrain_dir="output/"
dt=$(date '+%d-%m-%Y-%H-%M-%S');
out_dir=$pretrain_dir${dt}

echo "Time Signature: ${dt}"
echo "Saving Monolytic File Archive in : ${out_dir}/run_${dt}.txt"
if 
python  ./simclr_master/run.py --data_dir 'NIH/' \
--train_batch_size 16 \
--optimizer adam \
--model_dir $out_dir \
--checkpoint_path $out_dir \
--learning_rate 0.5 \
--use_blur \
--temperature 0.5 \
--proj_out_dim 128 \
--train_epochs 1 \
--train_data_split 0.0004 \
--checkpoint_epochs 20 \
--train_summary_steps 0 \
--color_jitter_strength 0.5 > run_${dt}.txt 2>&1;
then
echo "Time Signature:"$dt
echo "Saving Monolytic File Archive in : ${out_dir}/run_${dt}.txt"
cp run_${dt}.txt "${out_dir}/run_${dt}.txt"

cd $pretrain_dir
echo "PWD"
echo $PWD
tar -cvf $dt.tar.gz $dt
#mv $dt.tar.gz ../
fi
echo $dt
echo 'PreTraining Completed !!! '