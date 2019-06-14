export MXNET_CPU_WORKER_NTHREADS=56
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
logf="mobileface.log"
CUDA_VISIBLE_DEVICES='0,1,2,3' nohup python -u train.py \
     --network y1 \
     --loss softmax \
     --lr-steps="60000, 70000, 80000" \
     --dataset emore >  ./logs/$logf 2>&1 &

tail -f logs/$logf

