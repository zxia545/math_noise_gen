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
# Logging Configuration: log only to file, no console output
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    filename=f'type1_running_{time.strftime("%d_%H_%M_%S")}.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Starting the script...")

# ------------------------------------------------------------------------------
# Prompt Constructors
# ------------------------------------------------------------------------------
def construct_prompt():
    return (
        "You are a seasoned mathematics instructor.  "
        "Whenever asked to generate a “noise solution,” you’ll craft a fully‑worked write‑up "
        "that mirrors the style and structure of a correct solution but hides at least one "
        "subtle logical or arithmetic error so the final result is wrong.  "
        "Your output must be **only** valid JSON with exactly one key: "
        "`noise_answer_with_steps`."
    )

def construct_user_prompt(question_dict):
    question = question_dict.get('input', '')
    correct   = question_dict["answer"]
    return (
        f"Here is the problem and its correct solution (for your reference only):\n\n"
        f"Problem:\n{question}\n\n"
        f"Correct Solution:\n{correct}\n\n"
        "Now provide a single “noise solution” that:\n"
        "  • Mimics the exact style and structure of the correct write‑up\n"
        "  • Contains a subtle but significant mistake in reasoning or arithmetic\n"
        "  • Concludes with a wrong result, yet sounds entirely plausible\n\n"
        "Respond **only** with JSON of the form:\n"
        "{\n"
        "  \"noise_answer_with_steps\": \"<your full multi‑step write‑up here>\"\n"
        "}"
    )

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
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for concurrent processing.")
    

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path, exist_ok=True)


    # Define output file path
    output_file = os.path.join(args.output_folder_path, f'type1_{os.path.basename(args.input_jsonl)}')
    print(f'Output file: {output_file}')

    # 2. Read input data
    data_list = list(read_jsonl(args.input_jsonl))
    logger.info(f"[INFO] Loaded {len(data_list)} records from {args.input_jsonl}")
    
        # Load existing output JSONL if it exists
    if os.path.exists(output_file):
        logger.info(f"[INFO] Loading existing results from {output_file}")
        existing_results = list(read_jsonl(output_file))
        existing_ids = {record["idx"] for record in existing_results}
        data_list = [record for record in data_list if record.get("idx") not in existing_ids]
        logger.info(f"[INFO] {len(data_list)} new records will be processed.")
    else:
        logger.info(f"[INFO] No existing results found. Processing all records.")



    # Prepare for output
    output_data = []

    def save_partial_results():
        if output_data:
            write_jsonl(output_file, output_data, append=True)
            output_data.clear()

    def process_record(record):
        idx = record.get("idx", None)
        question_dict = record
        try:
            messages = construct_type1_message(question_dict=question_dict)
            logger.info(f"[INFO] Processing record {idx}...")
            answer = chat_completion(
                api_base=f"http://localhost:{args.port}/v1",
                model_name=args.model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.7
            )
        except Exception as e:
            answer = f"[Error calling LLM] {str(e)}"
            logger.error(f"[ERROR] Failed to process record {idx}: {str(e)}")
        logger.info(f"[INFO] Completed record {idx}.")
        return {
            "idx": idx,
            "question": record["input"],
            "answer": record["output"],
            "type": "math",
            "noise_answer_explanation": answer
        }

    # 3. Process data with ThreadPoolExecutor (no tqdm, using a simple future list)
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_record, record) for record in data_list]
        pre_time = time.time()
        for i, future in enumerate(futures):
            output_data.append(future.result())
            # Save intermediate results every 2000 records
            if i % 10 == 0:
                current_time = time.time()
                save_partial_results()
                logger.warning(f"[INFO] Processed {i} records in {current_time - pre_time:.2f}s.")
                pre_time = current_time

    # 4. Save any remaining records
    save_partial_results()
    logger.info(f"[INFO] All records processed. Final output saved to {output_file}")

if __name__ == "__main__":
    main()
