#!/usr/bin/env bash
GPU_MEM=32g
N_GPU=1
NGC_DATASET=82142
INSTANCE="dgx1v.${GPU_MEM}.${N_GPU}.norm"
DOCKER_IMAGE="nvidian/dlmed/clara-train-sdk:v4.0"
NOTE=".noninteractive"
#FOLDER_PATH="/workspace/Ali/BTCV/UNETR/BTCV_DDP"
FOLDER_PATH="/workspace/workspace/BTCV" # specify path
FOLD=0
VERSION=v1
BATCH_SIZE=1
FEATURE_SIZE=48
NAME="btcv.swinUNETR.epoch.fold${FOLD}.${VERSION}"
NAME_Job="btcv.fold${FOLD}.swinUNETR.${VERSION}.${N_GPU}x${GPU_MEM} ml-model.UNETR${NOTE}"

ngc batch run --name "${NAME_Job}" \
   --preempt RUNONCE --ace nv-us-west-2 --instance ${INSTANCE} \
   --result /results \
   --image ${DOCKER_IMAGE} \
   --org nvidian --team "dlmed" \
   --datasetid ${NGC_DATASET}:/dataset/dataset0\
   --workspace yucheng:/workspace:RW \
   --commandline "cd ${FOLDER_PATH} ; python __main__.py --batch_size=${BATCH_SIZE} --opt=adamw --num_steps=45000  --lrdecay --eval_num=100 --name=${NAME} --loss_type=dice_ce --conv_block --res_block --lr=1e-4 --ngc --fold=${FOLD}"
   
   

