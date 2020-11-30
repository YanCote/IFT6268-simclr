
echo ''
echo 'Starting task !'
dt=$(date '+%d-%m-%Y-%H-%M-%S');
root="/Users/yancote/mila/IFT6268-simclr/deployment"
echo 'Time Signature: ${dt}'
pretrain_dir="./${1:-yancote1}/"
mkdir $dt
mkdir $dt/archive
out_dir=$pretrain_dir$dt
echo $dt
echo $out_dir
echo $pretrain_dir
#echo "${out_dir}/run2_${dt}.txt"
echo 'PreTraining Completed !!! '
python pytorch-test.py > run_${dt}.txt
cp run_${dt}.txt  "${dt}/run2_${dt}.txt"
cp *.py "${dt}"
cp *.txt $root/$dt/archive

echo "$root/${dt}"
cd $root/$dt
tar -zcvf ../$dt.tar.gz .
cd ${dt}/archive
echo $PWD

