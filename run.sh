CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/home/cortex/vllm_bench python -m vllm_bench.benchmark \
    --trace /home/cortex/trace-single.jsonl \
    --result-filename /home/cortex/result-0.json \
    --save-result \
    --max-requests 2 \
    vllm-engine \
    --model /data-fast/s3/ml-dev-sfc-or-dev-misc1-k8s/yak/hf_models/mistralai/Codestral-22B-v0.1 \
    --tensor-parallel-size 1 \
    --disable-log-requests \
    --use-v2-block-manager \
    --max-num-seqs 8 \
    --max-model-len 8192 \
    --enforce-eager \
    --enable-chunked-prefill \
    --speculative-model '[ngram]' \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 4 \
