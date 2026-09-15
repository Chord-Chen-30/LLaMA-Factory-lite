# 2025-11-21 Impletentation of VL Deep Research Agent inference with tools

## Tools
Python, Image Search, Text Search, Visit


## Usage

1. Launch an vllm at http://localhost:8001/v1.
```bash
bash scripts/launch_vllm.sh
```

2. Modify the line in bash:
```python
python infer_w_tools.py --model_name_or_path [model_path] --data [jsonl_file_path] --output_dir [output_dir]
```
Output will be saved to [output_dir] named after original [file_name]_infer_result_w_tools.jsonl.

Support continue writing by setting `--overwrite 0`
