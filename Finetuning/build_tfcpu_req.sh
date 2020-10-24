module load python/3.7
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index tensorflow_cpu
pip install --no-index tensorflow_datasets
pip install --no-index tensorflow_hubs
pip freeze > buildrequirements.txt
deactivate
rm -rf $ENVDIR
