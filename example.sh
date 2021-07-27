train=$1
model=$2
bsz=$3
ddp=$4
ngpu_ddp=$5

# (DDP) or (nn.dataparallel, cpu)
if [ "${ddp}" = "ddp" ]
then
    cmd="${cmd}python -m torch.distributed.launch --nproc_per_node=${ngpu_ddp} --master_port=33133"
else
    cmd="${cmd}python"
fi


## model base setting
if [ "${model}" = "gmlp" ]
then
    layers="36"
    hidden="512"
    train_seq_len="512"
elif [ "${model}" = "amlp" ]
then
    layers="36"
    hidden="512"
    train_seq_len="512"
    attn_dim="64"    # tiny attention
elif [ "${model}" = "bert" ]
then
    layers="12"
    hidden="768"
    train_seq_len="512"
fi


if [ "${train}" = "train" ]
then
    cmd="${cmd} pretrain.py --model "${model}" -c data/train_data/ -v 35000 -o output/gmlp.model \
                --batch_size ${bsz}  --hidden ${hidden} --layers ${layers} --epochs 1000 --pretrain_seq_len ${train_seq_len}   \
                --lr 1e-4 "
elif [ "${train}" = "finetune" ]
then

    input_seq_len=50

    cmd="${cmd} finetune_nsmc.py --model "${model}" -c data/nsmc/ratings_train.txt -t data/nsmc/ratings_test.txt \
                -v 35000 -o output/gmlp.model \
                --batch_size ${bsz}  --hidden ${hidden} --layers ${layers} --epochs 50 --pretrain_seq_len ${train_seq_len}   \
                --lr 5e-5 --input_seq_len ${input_seq_len} --causal True"
fi



if [ "${ddp}" = "ddp" ]
then
    cmd="${cmd} --ddp True"
fi

if [ "${model}" = "amlp" ]   # add attention
then
    cmd="${cmd} --attn_dim ${attn_dim}"
fi



echo $cmd
$cmd
