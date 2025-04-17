import argparse
import os
import time
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import openai

# Local utility functions (replace with your own or adjust imports)
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion


# ------------------------------------------------------------------------------
# Main Execution 
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default="/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/0d6d11a6ea1187363aa7b78543f824fc02e06b14", 
        help="Path or name of the model for vLLM."
    )
    parser.add_argument("--model_name", type=str, default="filter_model", help="Name of the model for vLLM.")
    parser.add_argument("--output_folder_path", type=str, default="/home/aiscuser/zhengyu_blob_home/tony_folder/0413_math_with_noise/data/noise_data", help="Folder name for the output.")
    parser.add_argument("--gpu", type=int, default=8, help="Number of GPUs for tensor parallel.")
    parser.add_argument("--port", type=int, default=8020, help="Port for the vLLM server.")
    parser.add_argument("--input_jsonl", type=str, default="/home/aiscuser/zhengyu_blob_home/tony_folder/0413_math_with_noise/data/original_good_data/train_math.jsonl", help="Path to the input JSONL file.")
    parser.add_argument("--threads", type=int, default=60, help="Number of threads for concurrent processing.")

    args = parser.parse_args()


    # 1. Start vLLM server
    process = start_vllm_server(args.model, args.model_name, args.port, args.gpu)

    # wait until user control C is pressed
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping the server...")
        stop_vllm_server(process)
        sys.exit(0)

    stop_vllm_server(process)
    sys.exit(0)

if __name__ == "__main__":
    main()
