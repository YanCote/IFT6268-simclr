module load python/3.7
ENVDIR=~/scratch/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
diskusage_report
pip install --no-index tensorflow_cpu
pip install tensorflow_datasets --no-deps
pip install tensorflow_hub
diskusage_report
pip freeze > buildrequirements.txt
deactivate
rm -rf $ENVDIR
