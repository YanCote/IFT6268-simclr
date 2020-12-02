cp -u ~/build/* ~/models/pretrain
cd ~/models/pretrain
cat *.tar.gz | tar zxvf - -i
rm *.tar.gz