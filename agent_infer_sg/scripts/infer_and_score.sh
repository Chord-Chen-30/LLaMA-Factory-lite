# Inference + gpt-5-nano scoring for all checkpoint-400 models (plus the Qwen3-VL-30B-A3B-Thinking baseline).
# Run with: bash scripts/infer_and_score.sh
#
# For each model:
#   1. launch vLLM on port 8001 (TP=8)
#   2. wait for the server to be ready
#   3. run infer_w_tools.py over each dataset
#   4. kick off gpt-5-nano judge scoring in the background
#   5. kill vLLM and move on to the next model
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=python

# Make the conda env's libstdc++ (has CXXABI_1.3.15, needed by libicui18n.so.78)
# visible to the dynamic linker — we invoke ${PYTHON} directly instead of
# `conda activate`, which would otherwise set this for us.
export LD_LIBRARY_PATH="$(dirname "$(dirname "${PYTHON}")")/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

DATE=$(date +%Y-%m-%d)

if [ -z "${MODEL_PATHS+x}" ]; then
    MODEL_PATHS=(
        # "Qwen/Qwen3-VL-30B-A3B-Thinking"
        # "./saves/toolcall_3more_ocr_code_0421/checkpoint-400"
        # "./saves/qwen3_vl_30b_a8b_0421_critical_path/checkpoint-400" # Error <tool_response>
        # "./saves/qwen3_vl_30b_a8b_0421_optim_level3/checkpoint-400" # Error <tool_response>

        # ./saves/qwen3_vl_30b_a8b_0427_optim_level1/checkpoint-400 # Good
        # ./saves/qwen3_vl_30b_a8b_0427_optim_level2/checkpoint-400 # sub-optimal performance
        # ./saves/qwen3_vl_30b_a8b_0427_optim_level3/checkpoint-400 # sub-optimal performance

        # ./saves/qwen3_vl_30b_a8b_0429_optim_level2a/checkpoint-400
        # ./saves/qwen3_vl_30b_a8b_0429_optim_level3a/checkpoint-400

        # epochs=4 (lowest training loss)
        # ./saves/qwen3_vl_30b_a8b_0427_optim_level1/checkpoint-1396
        # ./saves/qwen3_vl_30b_a8b_0429_optim_level2a/checkpoint-1396
        # ./saves/qwen3_vl_30b_a8b_0429_optim_level3a/checkpoint-1396

        # 200 steps (lowest eval loss)
        ./saves/qwen3_vl_30b_a8b_0427_optim_level1/checkpoint-200
        ./saves/qwen3_vl_30b_a8b_0429_optim_level2a/checkpoint-200
        ./saves/qwen3_vl_30b_a8b_0429_optim_level3a/checkpoint-200
    )
fi

if [ -z "${DATASETS+x}" ]; then
    DATASETS=(
        "./agent_infer_sg/data/simplevqa.jsonl"
        "./agent_infer_sg/data/hle.jsonl"
        "./agent_infer_sg/data/mmsearch.jsonl"
        "./agent_infer_sg/data/livevqa.jsonl"
    )
fi

RESULT_ROOT=./agent_infer_sg/result

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "==== Launching vLLM for ${MODEL_PATH} ===="
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "${PYTHON}" -m vllm.entrypoints.openai.api_server \
        --allowed-local-media-path ./agent_infer_sg/data/ \
        --model "${MODEL_PATH}" \
        --tensor-parallel-size 8 \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --port 8001 &
    vllm_pid=$!

    # Wait for vLLM to be ready
    MAX_WAIT=1800
    for ((elapsed=0; elapsed<MAX_WAIT; elapsed+=10)); do
        if ! kill -0 "${vllm_pid}" 2>/dev/null; then
            echo "ERROR: vLLM exited before becoming ready" >&2
            exit 1
        fi
        if curl -sf "http://localhost:8001/v1/models" -o /dev/null; then
            echo "==== vLLM (port 8001) ready for ${MODEL_PATH} ===="
            break
        fi
        echo "Waiting for vLLM... (${elapsed}s)"
        sleep 10
    done

    if (( elapsed >= MAX_WAIT )); then
        echo "ERROR: vLLM not ready after ${MAX_WAIT}s" >&2
        kill "${vllm_pid}" 2>/dev/null
        exit 1
    fi

    # Server ready, run inference + scoring
    for dataset in "${DATASETS[@]}"; do
        echo "==== Inferring ${MODEL_PATH} on $(basename ${dataset}) ===="
        "${PYTHON}" src/infer_w_tools.py \
            --model_name_or_path "${MODEL_PATH}" \
            --data "${dataset}" \
            --date "${DATE}" \
            --mp 48 \
            --overwrite 0 \
            --log_level WARNING

        # Score in background so the next dataset's inference can start immediately.
        model_subdir=$(basename "$(dirname "${MODEL_PATH}")")/$(basename "${MODEL_PATH}")
        data_bn=$(basename "${dataset}" .jsonl)
        result_file="${RESULT_ROOT}/${DATE}/${model_subdir}/${data_bn}_w_tools.jsonl"
        echo "==== Scoring ${result_file} (background) ===="
        "${PYTHON}" src/gpt5_nano_score.py --data_path "${result_file}" --mp 8 --overwrite 1 &
    done

    echo "==== Shutting down vLLM (pid ${vllm_pid}) ===="
    if kill "${vllm_pid}" 2>/dev/null; then
        echo "Sent SIGTERM to vLLM (pid ${vllm_pid})"
    else
        echo "Warning: vLLM (pid ${vllm_pid}) was not running"
    fi
    # Give the server a moment to release GPUs before the next loop iteration.
    sleep 30
done

# Wait for any background scoring jobs to finish before exiting.
wait
echo "==== All done ===="