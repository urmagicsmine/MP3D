sleep 8h

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash tools/dist_train.sh \
configs/deeplesion/faster_rcnn_mp3d63_fpn_2x_deeplesion_fp16.py 8
