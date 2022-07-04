FILE=faster_rcnn_p3d63_fpn_2x_deeplesion_fp16_magzine
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_test.sh configs/deeplesion/$FILE.py ./work_dirs/deeplesion/$FILE/latest.pth 8 --out logs/1.pkl --eval bbox
