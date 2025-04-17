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
# Prompt Constructors
# ------------------------------------------------------------------------------
def construct_prompt():
    return (
        "You are given a math problem and its correct solution. "
        "that appears plausible but is ultimately incorrect. "
        "You must ensure the reasoning in your solution closely mimics a real solution's style and structure, "
        "but contains subtle errors or misleading steps so that the final conclusion is wrong.\n\n"
    )

def construct_user_prompt(question_dict):
    question = question_dict.get('input', '')
    correct_answer = question_dict.get('answer', '')
    # correct_explanation = question_dict.get('explanation', '')

    prompt = (
        f"Math Problem:\n{question}\n\n"
        f"Correct Solution (for reference only):\n{correct_answer}\n\n"
        "Now provide a single 'noise solution' that looks similar in structure to the correct solution, "
        "but is incorrect. Do NOT simply restate the correct solution. "
        "You must introduce at least one significant error in the calculation or reasoning.\n\n"
        "Provide a plausible but incorrect 'answer' and a misleading 'explanation' to generate ambiguity for this question. "
        "Your response must ONLY include the 'answer' and 'explanation' keys formatted as a dictionary.\n\n"
        "Example:\n"
        "{\n"
        "  \"answer\": \"your answer\",\n"
        "  \"explanation\": \"your explanation\"\n"
        "}"
    )
    return prompt

def construct_type1_message(question_dict):
    """
    Constructs the message structure for Type1 interaction.
    """
    s_prompt = construct_prompt()
    prompt = construct_user_prompt(question_dict=question_dict)
    return [
        {"role": "system", "content": s_prompt},
        {"role": "user", "content": prompt}
    ]

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


    # 2. Read input data
    data_list = list(read_jsonl(args.input_jsonl))


    def process_record(record):
        idx = record.get("idx", None)
        question_dict = record
        try:
            messages = construct_type1_message(question_dict=question_dict)
            print(f"[INFO] Processing record {idx}...")
            answer = chat_completion(
                api_base=f"http://localhost:{args.port}/v1",
                model_name=args.model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.7
            )
            print(f"[INFO] Answer for record {idx}: {answer}")
        except Exception as e:
            answer = f"[Error calling LLM] {str(e)}"
            print(f"[ERROR] Failed to process record {idx}: {str(e)}")
        print(f"[INFO] Completed record {idx}.")
        return 0

    # 3. Process data with ThreadPoolExecutor (no tqdm, using a simple future list)
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        for _ in range(100):
            futures = [executor.submit(process_record, record) for record in data_list] 

if __name__ == "__main__":
    main()
