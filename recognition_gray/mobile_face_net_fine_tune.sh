export MXNET_CPU_WORKER_NTHREADS=56
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
logf="mobileface.log"
CUDA_VISIBLE_DEVICES='0,1,2,3' nohup python -u train.py \
     --network y1 \
     --loss triplet \
     --lr 0.005 \
     --pretrained ./models/y1-softmax-emore/model  >  ./logs/$logf 2>&1 &

tail -f logs/$logf

