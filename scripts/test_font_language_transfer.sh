#!/bin/bash -f

#=====================================
# MC-GAN
# Train and Test conditional GAN Glyph network
# By Samaneh Azadi
# Updated 11/21/19 Sam Goree
#=====================================


#=====================================
## Set Parameters
#=====================================

# DATA=$1
DATASET="./datasets/Capitals64_resized"
# DATASET="./datasets/devangari_fonts/full_fonts/"
experiment_dir="GlyphNet_train_devangari_auxiliary"
AUXILIARYDATAROOT="./datasets/Capitals64_resized/"
AUXILIARYMISSINGCHARACTERS=44
MODEL=cGAN
MODEL_G=resnet_6blocks
MODEL_D=n_layers
n_layers_D=1
NORM=batch
IN_NC=70
O_NC=70
GRP=70
PRENET=none # change back to 2_layers
FINESIZE=32
LOADSIZE=32
LAM_A=100
NITER=250
NITERD=100
BATCHSIZE=150
EPOCH=latest #test at which epoch?
CUDA_ID=0

if [ ! -d "./checkpoints/${experiment_dir}" ]; then
	mkdir "./checkpoints/${experiment_dir}"
fi
LOG="./checkpoints/${experiment_dir}/test.txt"
if [ -f $LOG ]; then
	rm $LOG
fi
exec &> >(tee -a "$LOG")


# =======================================
## Test Glyph Network on font dataset
# =======================================
# TODO: Add back --conditional
CUDA_VISIBLE_DEVICES=${CUDA_ID} python test.py --dataroot ${DATASET} --name "${experiment_dir}"\
							 	--model ${MODEL} --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} --which_model_preNet ${PRENET}\
							 	--norm ${NORM} --input_nc ${IN_NC} --output_nc ${O_NC} --grps ${GRP}  --loadSize ${FINESIZE} --fineSize ${LOADSIZE} --display_id 0 --batchSize 1 \
							 	--which_epoch ${EPOCH} --blanks 0.8 --conv3d --align_data --onlyconditionononelanguage



