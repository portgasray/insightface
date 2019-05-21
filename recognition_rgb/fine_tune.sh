export MXNET_CPU_WORKER_NTHREADS=56
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
logf="test_fine_tune_gray.log"
CUDA_VISIBLE_DEVICES='0,1,2,3' nohup python -u train.py \
     --network m1 \
     --loss triplet \
     --lr 0.005 \
     --pretrained ./models/m1-softmax-emore/model > ./logs/$logf 2>&1 &

tail -f logs/$logf
