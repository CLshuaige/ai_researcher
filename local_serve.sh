
vllm serve \
  /home/ai_researcher/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8001 \
  --enable-auto-tool-choice \
  --tool-call-parser pythonic

