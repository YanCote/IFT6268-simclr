rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*"  ${1:-yancote1}@graham.calculcanada.ca:/home/${1:-yancote1}/models/pretrain/ ./models/pretrain
rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*"  ${1:-yancote1}@cedar.calculcanada.ca:/home/${1:-yancote1}/models/pretrain/ ./models/pretrain
rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*"  ${1:-yancote1}@beluga.calculcanada.ca:/home/${1:-yancote1}/models/pretrain/ ./models/pretrain

#rsync -zarvu --prune-empty-dirs --include "*/" --include="*.gz" --exclude="*" ./models/pretrain/build ${1:-yancote1}@graham.calculcanada.ca:/home/${1:-yancote1}/build