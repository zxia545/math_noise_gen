import argparse
import os
import time
import json
import logging
import sys
import re
from concurrent.futures import ThreadPoolExecutor

import openai # Assuming chat_completion uses the openai library structure

# --- Assuming these utilities exist from your previous context ---
# You might need to adjust paths or copy these functions here
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion
# --- End assumed utilities ---

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
# Create a unique log file name including 'validate'
log_filename = f'validate_regenerate_running_{time.strftime("%d_%H_%M_%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    filename=log_filename,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Also log to console for interactive feedback during validation runs
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info(f"Starting the validation and regeneration script. Logging to {log_filename}")

# ------------------------------------------------------------------------------
# Helper Function to Extract JSON from LLM Output
# ------------------------------------------------------------------------------
def extract_json_from_string(text):
    """
    Extracts the first valid JSON object string, handling potential markdown code blocks.
    Returns the parsed JSON object or None if no valid JSON is found.
    """
    # Look for JSON within markdown code blocks first
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:
        # Look for the first '{' and the last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
        else:
            logger.warning(f"Could not find JSON structure in text: {text[:100]}...")
            return None # No JSON structure found

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to decode JSON: {e}\nJSON string was: {json_str}")
        return None
    

# ------------------------------------------------------------------------------
# Prompt Constructors (Original Noise Generation) - Reused from your script
# ------------------------------------------------------------------------------
def construct_noise_gen_system_prompt():
    return (
        "You are a seasoned mathematics instructor. "
        "Whenever asked to generate a “noise solution,” you’ll craft a fully‑worked write‑up "
        "that mirrors the style and structure of a correct solution but hides at least one "
        "subtle logical or arithmetic error so the final result is wrong. "
        "Your output must be **only** valid JSON with exactly one key: `noise_answer_with_steps`.\n\n"
        "IMPORTANT: The final numeric result you provide must NOT equal the reference answer, "
        "and you must avoid re‑using any of the same numbers or expressions from the correct solution."
    )

def construct_noise_gen_user_prompt(question_dict):
    question = question_dict.get('question', '') # Adjusted key based on typical structure
    correct = question_dict.get("answer", '')   # Adjusted key
    return (
        f"Here is the problem and its correct solution (for your reference only):\n\n"
        f"Problem:\n{question}\n\n"
        f"Correct Solution:\n{correct}\n\n"
        "Now provide a single “noise solution” that:\n"
        "  • Mimics the exact style and structure of the correct write‑up\n"
        "  • Contains a subtle but significant mistake in reasoning or arithmetic\n"
        "  • Concludes with a wrong result, yet sounds entirely plausible\n"
        "  • **Does not** use any of the same numeric values or algebraic expressions as the correct solution\n\n"
        "Respond **only** with JSON of the form:\n"
        "{\n"
        "  \"noise_answer_with_steps\": \"<your full multi‑step write‑up here>\"\n"
        "}"
    )

def construct_noise_gen_message(question_dict):
    s_prompt = construct_noise_gen_system_prompt()
    prompt = construct_noise_gen_user_prompt(question_dict=question_dict)
    return [
        {"role": "system", "content": s_prompt},
        {"role": "user", "content": prompt}
    ]

# ------------------------------------------------------------------------------
# Prompt Constructors (Noise Validation) - NEW
# ------------------------------------------------------------------------------
def construct_validation_system_prompt():
    return (
        "You are a meticulous Math Problem Verifier. Your task is to evaluate a proposed 'noise solution' "
        "against the original problem and its known correct solution. A 'noise solution' is intended to look "
        "plausible but contain a subtle error leading to an incorrect final answer, without directly copying "
        "elements from the correct solution."
        "Your goal is to determine if the provided 'noise solution' successfully meets these criteria."
        "Respond **only** with valid JSON containing two keys: `is_valid_noise` (boolean) and `reason` (string)."
    )

def construct_validation_user_prompt(question, correct_solution, noise_solution_text):
    return (
        f"Please evaluate the following 'Noise Solution' based on the provided Problem and Correct Solution.\n\n"
        f"Problem:\n{question}\n\n"
        f"Correct Solution:\n{correct_solution}\n\n"
        f"Proposed Noise Solution:\n{noise_solution_text}\n\n"
        f"Verification Criteria:\n"
        f"1. Does the noise solution present a step-by-step derivation that seems plausible?\n"
        f"2. Does it contain at least one identifiable (even if subtle) logical or arithmetic error?\n"
        f"3. Is the **final numeric result** of the noise solution different from the final result of the correct solution?\n"
        f"4. Does the noise solution avoid significant reuse of specific numeric values or algebraic expressions from the correct solution (beyond standard mathematical constants or operators)?\n"
        f"5. Does the noise solution mimic the overall style and structure of the correct solution?\n\n"
        f"Determine if the 'Proposed Noise Solution' is **valid noise**. It is **invalid** if:\n"
        f"  a) It arrives at the correct final answer.\n"
        f"  b) It is substantially identical to the correct solution.\n"
        f"  c) It contains no clear error or the derivation is nonsensical.\n"
        f"  d) It heavily copies numerical values or expressions from the correct solution.\n"
        f"  e) It fails to mimic the style/structure appropriately.\n\n"
        f"Respond **only** with a JSON object indicating validity and the reason:\n"
        f"{{\n"
        f'  "is_valid_noise": <true_or_false>,\n'
        f'  "reason": "<Brief explanation for your decision, referencing the criteria>"\n'
        f"}}"
    )

def construct_validation_message(question, correct_solution, noise_solution_text):
    s_prompt = construct_validation_system_prompt()
    prompt = construct_validation_user_prompt(question, correct_solution, noise_solution_text)
    return [
        {"role": "system", "content": s_prompt},
        {"role": "user", "content": prompt}
    ]

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate and regenerate noise answers in a JSONL file.")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to the input JSONL file containing potentially noisy data."
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="./validated_noise_data",
        help="Folder name for the output validated/regenerated data."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/path/to/your/model", # <<< IMPORTANT: Set a default or require this
        help="Path or name of the model for vLLM (used for both validation and regeneration)."
    )
    parser.add_argument("--model_name", type=str, default="validator_regenerator_model", help="Name of the model for vLLM.")
    parser.add_argument("--gpu", type=int, default=8, help="Number of GPUs for tensor parallel.")
    parser.add_argument("--port", type=int, default=8020, help="Port for the vLLM server.")
    parser.add_argument("--threads", type=int, default=30, help="Number of threads for concurrent processing (adjust based on GPU memory and validation complexity).")
    parser.add_argument("--max_tokens_validate", type=int, default=512, help="Max tokens for the validation response.")
    parser.add_argument("--max_tokens_regenerate", type=int, default=2048, help="Max tokens for regenerated noise.")
    parser.add_argument("--temperature_validate", type=float, default=0.1, help="Temperature for validation (low for consistency).")
    parser.add_argument("--temperature_regenerate", type=float, default=0.7, help="Temperature for regeneration (higher for variability).")
    parser.add_argument("--save_interval", type=int, default=500, help="Save progress every N records.")

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path, exist_ok=True)
        logger.info(f"Created output directory: {args.output_folder_path}")

    # Define output file path
    base_name = os.path.basename(args.input_jsonl)
    output_file = os.path.join(args.output_folder_path, f'validated_{base_name}')
    logger.info(f'Input file: {args.input_jsonl}')
    logger.info(f'Output file: {output_file}')

    # --- Check if output file already exists and handle resume ---
    processed_indices = set()
    if os.path.exists(output_file):
        logger.warning(f"Output file {output_file} already exists. Attempting to resume.")
        try:
            # Read existing output to find which indices are already processed
            existing_data = list(read_jsonl(output_file))
            processed_indices = {record.get("idx") for record in existing_data if "idx" in record}
            logger.info(f"Resuming. Found {len(processed_indices)} already processed records.")
            # Keep the existing data in memory to append new results later (if file is not too large)
            # For very large files, it might be better to always append and handle duplicates post-hoc
            output_data_list = existing_data
        except Exception as e:
            logger.error(f"Error reading existing output file for resuming: {e}. Starting fresh.", exc_info=True)
            output_data_list = [] # Start fresh if reading fails
            # Optionally backup the old file here
            # os.rename(output_file, output_file + f".backup_{time.strftime('%Y%m%d_%H%M%S')}")
    else:
        output_data_list = []

    # --- Read input data ---
    try:
        input_data_list = list(read_jsonl(args.input_jsonl))
        logger.info(f"Loaded {len(input_data_list)} records from {args.input_jsonl}")
    except Exception as e:
        logger.critical(f"Failed to load input file {args.input_jsonl}: {e}", exc_info=True)
        sys.exit(1) # Exit if input can't be loaded

    # Filter out records that are already processed
    records_to_process = [record for record in input_data_list if record.get("idx") not in processed_indices]
    logger.info(f"Will process {len(records_to_process)} new records.")

    if not records_to_process:
        logger.info("No new records to process. Exiting.")
        sys.exit(0)

    # --- Start vLLM server ---
    logger.info("Starting vLLM server...")
    process = start_vllm_server(args.model, args.model_name, args.port, args.gpu)
    if not process:
         logger.critical("Failed to start vLLM server. Exiting.")
         sys.exit(1) # Exit if server fails to start

    # --- Processing Function ---
    def process_record(record):
        idx = record.get("idx", "N/A")
        question = record.get("question") or record.get("input") # Handle potential key variation
        correct_answer = record.get("answer") or record.get("output") # Handle potential key variation
        original_noise_payload = record.get("noise_answer_explanation")

        if not all([question, correct_answer, original_noise_payload]):
            logger.warning(f"Record {idx}: Missing required fields (question, answer, or noise_answer_explanation). Skipping.")
            # Return original record structure maybe with an error note
            return {**record, "validation_status": "skipped_missing_fields", "noise_needs_regeneration": True}

        validation_result = {"is_valid_noise": False, "reason": "Validation not performed"}
        noise_to_use = original_noise_payload
        regenerated = False

        # 1. Extract the actual noise text from the payload (which might be JSON)
        original_noise_json = extract_json_from_string(original_noise_payload)
        if original_noise_json and 'noise_answer_with_steps' in original_noise_json:
            noise_text = original_noise_json['noise_answer_with_steps']
        else:
            # If it's not the expected JSON, treat the whole string as the noise (less ideal)
            # Or force regeneration if structure is mandatory
            logger.warning(f"Record {idx}: Could not parse expected JSON from noise_answer_explanation. Treating raw string as noise for validation / forcing regeneration.")
            noise_text = str(original_noise_payload) # Ensure it's a string
            # Decide: force regeneration if JSON structure was required by original script?
            # validation_result = {"is_valid_noise": False, "reason": "Failed to parse noise JSON structure."}

        # 2. Validate the extracted/original noise text
        try:
            logger.debug(f"Record {idx}: Validating noise.")
            validation_messages = construct_validation_message(question, correct_answer, noise_text)
            validation_response_raw = chat_completion(
                api_base=f"http://localhost:{args.port}/v1",
                model_name=args.model_name,
                messages=validation_messages,
                max_tokens=args.max_tokens_validate,
                temperature=args.temperature_validate
            )
            validation_response_json = extract_json_from_string(validation_response_raw)

            if validation_response_json and isinstance(validation_response_json.get("is_valid_noise"), bool):
                validation_result = validation_response_json
                logger.info(f"Record {idx}: Validation result: {validation_result['is_valid_noise']}. Reason: {validation_result['reason']}")
            else:
                logger.error(f"Record {idx}: Validation LLM response was not valid JSON or missing 'is_valid_noise'. Response: {validation_response_raw[:200]}...")
                validation_result = {"is_valid_noise": False, "reason": "Validation LLM failed or gave malformed response."}

        except Exception as e:
            logger.error(f"Record {idx}: Error during validation LLM call: {e}", exc_info=True)
            validation_result = {"is_valid_noise": False, "reason": f"Validation Error: {str(e)}"}

        # 3. Regenerate if validation failed
        if not validation_result.get("is_valid_noise"):
            logger.warning(f"Record {idx}: Noise validation failed or deemed invalid. Regenerating noise.")
            regenerated = True
            try:
                # Construct the prompt using original keys expected by generation function
                gen_question_dict = {"question": question, "answer": correct_answer}
                regeneration_messages = construct_noise_gen_message(gen_question_dict)
                new_noise_payload = chat_completion(
                    api_base=f"http://localhost:{args.port}/v1",
                    model_name=args.model_name,
                    messages=regeneration_messages,
                    max_tokens=args.max_tokens_regenerate,
                    temperature=args.temperature_regenerate
                )
                # Basic check on regenerated payload
                if not new_noise_payload or not extract_json_from_string(new_noise_payload):
                     logger.error(f"Record {idx}: Regeneration failed to produce valid JSON payload. Keeping original invalid noise.")
                     # Keep original noise but mark as failed regeneration
                     validation_result["reason"] += " | Regeneration attempt failed."
                else:
                    noise_to_use = new_noise_payload # Use the newly generated noise
                    logger.info(f"Record {idx}: Successfully regenerated noise.")
                    # Optionally: Re-validate the regenerated noise (could double cost/time)
                    # logger.info(f"Record {idx}: Re-validating regenerated noise...")
                    # ... re-validation logic ...

            except Exception as e:
                logger.error(f"Record {idx}: Error during regeneration LLM call: {e}", exc_info=True)
                # Keep original noise, but add note about regen failure
                validation_result["reason"] += f" | Regeneration Error: {str(e)}"


        # 4. Construct final output record
        output_record = {
            **record, # Keep all original fields
            "noise_answer_explanation": noise_to_use, # Use original or regenerated
            "validation_passed": validation_result.get("is_valid_noise", False),
            "validation_reason": validation_result.get("reason", "Unknown"),
            "noise_was_regenerated": regenerated
        }
        return output_record

    # --- Process data with ThreadPoolExecutor ---
    processed_count = 0
    last_save_count = 0
    start_time = time.time()

    # Temporary list to hold results between saves
    current_batch_results = []

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit tasks
        futures = {executor.submit(process_record, record): record.get("idx") for record in records_to_process}

        for future in futures: # Iterate through submitted futures
            idx = futures[future]
            try:
                result = future.result()
                current_batch_results.append(result)
                processed_count += 1

                # Log progress periodically
                if processed_count % 50 == 0: # Log every 50 records
                     elapsed_time = time.time() - start_time
                     rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                     logger.info(f"Progress: Processed {processed_count}/{len(records_to_process)} records. Rate: {rate:.2f} rec/s.")

                # Save intermediate results periodically
                if processed_count - last_save_count >= args.save_interval:
                    logger.warning(f"Saving intermediate results ({len(current_batch_results)} records processed in this batch)...")
                    write_jsonl(output_file, current_batch_results, append=True) # Append to the file
                    output_data_list.extend(current_batch_results) # Add to in-memory list if resuming correctly
                    current_batch_results = [] # Clear the batch
                    last_save_count = processed_count
                    logger.info(f"Intermediate save complete. Total processed: {processed_count}")

            except Exception as e:
                logger.error(f"Record {idx}: Critical error processing future: {e}", exc_info=True)
                # Optionally save a placeholder error record
                current_batch_results.append({
                    "idx": idx,
                    "error": f"Failed during processing: {str(e)}",
                    **records_to_process[processed_count] # Include original data if possible
                 })
                processed_count += 1 # Ensure count increments even on error


    # --- Save any remaining records ---
    if current_batch_results:
        logger.info(f"Saving final batch of {len(current_batch_results)} records...")
        write_jsonl(output_file, current_batch_results, append=True)
        # output_data_list.extend(current_batch_results) # Add final batch to in-memory list
        logger.info(f"Final save complete.")

    total_time = time.time() - start_time
    logger.info(f"All {len(records_to_process)} requested records processed in {total_time:.2f} seconds.")
    logger.info(f"Final validated output saved to {output_file}")

    # --- Stop the vLLM server ---
    logger.info("Stopping vLLM server...")
    stop_vllm_server(process)
    logger.info("Script finished.")


if __name__ == "__main__":
    main()