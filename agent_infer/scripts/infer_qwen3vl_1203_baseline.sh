# MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Thinking"

# 暂时修改，cz
MODEL_PATH="./saves/qwen3_vl_30b_a8b_1203_cleaned/checkpoint-100"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
--allowed-local-media-path ./images \
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
    "./eval_data/simplevqa_300.jsonl"
    # "./eval_data/hle_gaia_mm_v1.jsonl"
    # "./eval_data/mmsearch_171.jsonl"
    # "./eval_data/livevqa_300.jsonl"   
)

cd ./agent_infer

for dataset in "${datasets[@]}"; do
    python src/infer_w_tools_mp_test.py --model_name_or_path ${MODEL_PATH} --data ${dataset} --date 2025-12-04
done



# echo "==== 关闭服务... ===="
if kill ${vllm_pid}; then
    echo "成功关闭VLLM服务 (PID: ${vllm_pid})"
else
    echo "警告：未能关闭VLLM服务 (PID: ${vllm_pid})，可能已被关闭或不存在。"
fi

# score
cd ./agent_infer/src;
python qwen_max_score.py --data_path ../result/2025-12-04/checkpoint-100/simplevqa_300_w_tools.jsonl