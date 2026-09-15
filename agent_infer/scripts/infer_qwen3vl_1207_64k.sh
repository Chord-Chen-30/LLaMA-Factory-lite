# Wait PID
PID=790059
while ps -p "$PID" >/dev/null 2>&1; do
    sleep 30
    echo "Waiting for PID $PID to exit..."
done
echo "Process $PID has exited."


MODEL_PATHS=(
    "Qwen/Qwen3-VL-30B-A3B-Thinking"
    "./saves/qwen3_vl_30b_a8b_1206_64k/checkpoint-174" # 2 epochs
    "./saves/qwen3_vl_30b_a8b_1206_64k/checkpoint-348" # 4 epochs
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
--allowed-local-media-path ./agent_infer/data/ \
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


datasets=(
    # "./eval_data/simplevqa_300.jsonl"
    "./agent_infer/data/hle.jsonl"
    # "./eval_data/mmsearch_171.jsonl"
    # "./eval_data/livevqa_300.jsonl"   
)

cd ./agent_infer

for dataset in "${datasets[@]}"; do
    python src/infer_w_tools_mp_test.py --model_name_or_path ${MODEL_PATH} --data ${dataset} --date 2025-12-07 --mp 8 --overwrite 0

    model_bn=$(basename "$MODEL_PATH")
    data_bn=$(basename "$dataset")
    data_bn=${data_bn%.jsonl}
    python src/qwen_max_score.py --data_path ./result/2025-12-07/${model_bn}/${data_bn}_w_tools.jsonl --overwrite 1 &
done


echo "==== 关闭服务... ===="
if kill ${vllm_pid}; then
    echo "成功关闭VLLM服务 (PID: ${vllm_pid})"
else
    echo "警告：未能关闭VLLM服务 (PID: ${vllm_pid})，可能已被关闭或不存在。"
fi

done

wait