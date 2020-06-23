export MXNET_CPU_WORKER_NTHREADS=56
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
logf="test_train_gray.log"
CUDA_VISIBLE_DEVICES='' nohup python -u train.py \
     --network m1 \
     --loss softmax \
     --lr-steps="200000, 320000, 440000" \
     --dataset emore > ./logs/$logf 2>&1 &

tail -f logs/$logf

