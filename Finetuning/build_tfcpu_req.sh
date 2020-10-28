module load python/3.7
ENVDIR=~/scratch/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install setuptools
pip download --no-deps tensorflow_datasets
# # pip install --no-index promise
# #pip install --no-index tensorflow_datasets
# pip install --no-index ~/python_packages/promise-2.2.1-py2-none-any.whl
# pip install --no-index absl-py
# pip install --no-index ~/python_packages/tensorflow_metadata-0.24.0-py3-none-any.whl
# pip install --no-index ~/python_packages/tensorflow_datasets-4.0.1-py3-none-any.whl
# # pip install --no-index tensorflow_gpu
# # pip install --no-index tensorflow_hub --no-deps
pip freeze > buildrequirements.txt
deactivate
# rm -rf $ENVDIRcd ~/pyth   