
echo ''
echo 'Starting task !'
dt=$(date '+%d-%m-%Y-%H-%M-%S');

echo 'Time Signature: ${dt}'
pretrain_dir="./${1:-yancote1}/"
mkdir $pretrain_dir
out_dir=$pretrain_dir$dt
echo $dt
echo $out_dir
echo {$out_dir}
echo "${out_dir}/run2_${dt}.txt"
echo 'PreTraining Completed !!! '
python pytorch-test.py > run_${dt}.txt
cp run_${dt}.txt  "${pretrain_dir}/run2_${dt}.txt"
