MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 500 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 1"
NUM_GPUS=2


# mpiexec -n $NUM_GPUS \
python scripts/image_train.py --data_dir /home2/shyammarjit/diseg/segformer/data/ade/ADEChallengeData2016/images/train\
    $MODEL_FLAGS\
    $DIFFUSION_FLAGS\
    $TRAIN_FLAGS