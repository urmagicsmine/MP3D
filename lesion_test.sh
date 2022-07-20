MODELNAME=$1
CUDA_VISIBLE_DEVICES=1,2,3,5 tools/dist_test.sh \
        configs/deeplesion/$MODELNAME.py \
        work_dirs/$MODELNAME/latest.pth 4 \
        --eval bbox \
