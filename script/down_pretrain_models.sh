# Executed in hard coded path
# LOCAL BUILD are in ./models/pretrain/build
# REMOTE BUILD are in /home/build
#E Execute from projet root directory
# args1 : cedar beluga or graham: download .tar.gx from the server, if upload, upload all build to graham
# args2 : user name

if [ "$1" = "cedar" ]
then
  rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*"  ${2:-yancote1}@cedar.calculcanada.ca:/home/${2:-yancote1}/models/pretrain/ ./models/pretrain
fi
if [ "$1" = "graham" ]
then
  rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*"  ${2:-yancote1}@graham.calculcanada.ca:/home/${2:-yancote1}/models/pretrain/ ./models/pretrain
fi
if [ "$1" = "beluga" ]
then
  rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*"  ${2:-yancote1}@beluga.calculcanada.ca:/home/${2:-yancote1}/models/pretrain/ ./models/pretrain
fi
if [ "$1" = "upload" ]
then
  rsync -a script/extract_build.sh ${2:-yancote1}@graham.calculcanada.ca:/home/${2:-yancote1}/build
  rsync -a script/extract_build.sh ${2:-yancote1}@beluga.calculcanada.ca:/home/${2:-yancote1}/build
  rsync -a script/extract_build.sh ${2:-yancote1}@cedar.calculcanada.ca:/home/${2:-yancote1}/build
  rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*" ./models/pretrain/build ${2:-yancote1}@graham.calculcanada.ca:/home/${2:-yancote1}
  rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*" ./models/pretrain/build ${2:-yancote1}@beluga.calculcanada.ca:/home/${2:-yancote1}
  rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*" ./models/pretrain/build ${2:-yancote1}@cedar.calculcanada.ca:/home/${2:-yancote1}

fi
