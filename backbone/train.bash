output_dir="./exp/"

# PISR dataset
root_dir=''
list="StandingRabbit LyingRabbit"
for i in $list; do
    time=$(date "+%Y%m%d%H%M%S")
    python train.py \
        -s ${root_dir}${i} \
        -m ${output_dir}${i}_${time}  \
        --resolution 2 \
        --iterations 15000 \
        --stokes_l 0.1 \
        --azimuth_l 0.05 \
        --azimuth_start 1000\
    python render.py \
    -m ${output_dir}${i}_${time} \
    --resolution 2 \
    --img \
    --iteration -1
done

# SMVP3D dataset
root_dir=''
list="snail david hedgehog dragon squirrel"

for i in $list; do
    time=$(date "+%Y%m%d%H%M%S")
    python train.py \
        -s ${root_dir}${i} \
        -m ${output_dir}${i}_${time}  \
        --resolution 1 \
        --iterations 15000 \
        --stokes_l 0.1 \
        --azimuth_l 0.5 \
        --azimuth_start 1000
    python render.py \
    -m ${output_dir}${i}_${time} \
    --resolution 1 \
    --img \
    --iteration -1
done

# RMVP3D dataset
root_dir=''
list="shisa frog dog"
for i in $list; do
    time=$(date "+%Y%m%d%H%M%S")
    python train.py \
        -s ${root_dir}${i} \
        -m ${output_dir}${i}_${time}  \
        --resolution 2 \
        --iterations 15000 \
        --stokes_l 0.1\
        --azimuth_l 0.1 \
        --azimuth_start 1000\
    python render.py \
    -m ${output_dir}${i}_${time} \
    --resolution 2 \
    --img \
    --iteration -1
done

# PANDORA dataset
list="owl vase"
for i in $list; do
    time=$(date "+%Y%m%d%H%M%S")
    python train.py \
        -s ${root_dir}${i} \
        -m ${output_dir}${i}_${time}  \
        --resolution 4 \
        --iterations 15000 \
        --stokes_l 0.1 \
        --azimuth_l 0.1 \
        --azimuth_start 1000\

    python render.py \
    -m ${output_dir}${i}_${time} \
    --resolution 4 \
    --img \
    --iteration -1

done