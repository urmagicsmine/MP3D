MODELNAME=$1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/dist_train.sh \
configs/deeplesion/$MODELNAME 4
