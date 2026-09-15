
# MODEL_PATH="./saves/qwen3_vl_30b_a3b_1022_lr1e-5"
MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Thinking"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
--allowed-local-media-path ./agent_infer/temp/eval_data/images \
--model ${MODEL_PATH} \
--tensor-parallel-size 8 \
--dtype bfloat16 \
--host 0.0.0.0 \
--port 8001 & vllm_pid=$!


while true; do
    if curl -s http://localhost:8001/v1/chat/completions > /dev/null; then
        echo "Local model (port 8001) is ready!"
        break
    fi
    echo "Waiting for servers to start ....."
    sleep 30
done

echo "==== ${MODEL_PATH} Loaded. Infer ... ===="

cd ./agent_infer


datasets=(
    "./agent_infer/temp/eval_data/mix/mix.jsonl" 
)

for dataset in "${datasets[@]}"; do
    python src/infer_w_tools.py --model_name_or_path ${MODEL_PATH} --data ${dataset} --overwrite 0
done




# echo "==== 关闭服务... ===="
if kill ${vllm_pid}; then
    echo "成功关闭VLLM服务 (PID: ${vllm_pid})"
else
    echo "警告：未能关闭VLLM服务 (PID: ${vllm_pid})，可能已被关闭或不存在。"
fi