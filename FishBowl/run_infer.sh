CUDA_VISIBLE_DEVICES=1

TRAIN_METHOD="bidirectional_consist_next_vm_label_1.5bbox_finalconsist"

python -m torch.distributed.launch --nproc_per_node=1 \
main.py --mode test --training_method ${TRAIN_METHOD} \
--log_path log_${TRAIN_METHOD} --device cuda --batch_size 1 \
--data_path "" --num_workers 1 --loss_type BCE \
--enlarge_coef 1.5