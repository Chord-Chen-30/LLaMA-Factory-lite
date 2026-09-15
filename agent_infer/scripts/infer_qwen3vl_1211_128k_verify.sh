# Wait PID
PID=790059
while ps -p "$PID" >/dev/null 2>&1; do
    sleep 30
    echo "Waiting for PID $PID to exit..."
done
echo "Process $PID has exited."

# 文件存在后再执行
FILE_PATH="./saves/qwen3_vl_30b_a8b_1211_128k_verify/all_results.json"

# 循环检查文件是否存在
while true; do
    if [[ -f "$FILE_PATH" ]]; then
        echo "文件已存在: $FILE_PATH"
        break
    else
        echo "文件不存在，等待1分钟后重试..."
        sleep 60
    fi
done
echo "文件检测成功，继续执行后续操作..."


MODEL_PATHS=(
    # "./saves/qwen3_vl_30b_a8b_1211_128k_verify/checkpoint-348" # 4 epochs
    "./saves/qwen3_vl_30b_a8b_1211_128k_verify/checkpoint-150" # lowest dev
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
    "./agent_infer/data/simplevqa.jsonl"
    "./agent_infer/data/hle.jsonl"
    "./agent_infer/data/mmsearch.jsonl"
    "./agent_infer/data/livevqa.jsonl"
)

cd ./agent_infer

for dataset in "${datasets[@]}"; do
    python src/infer_w_tools_mp_test.py --model_name_or_path ${MODEL_PATH} --data ${dataset} --date 2025-12-11 --mp 12 --overwrite 1

    model_bn=$(basename "$MODEL_PATH")
    data_bn=$(basename "$dataset")
    data_bn=${data_bn%.jsonl}
    python src/qwen_max_score.py --data_path ./result/2025-12-11/${model_bn}/${data_bn}_w_tools.jsonl --overwrite 1 &
done


echo "==== 关闭服务... ===="
if kill ${vllm_pid}; then
    echo "成功关闭VLLM服务 (PID: ${vllm_pid})"
else
    echo "警告：未能关闭VLLM服务 (PID: ${vllm_pid})，可能已被关闭或不存在。"
fi

done

wait