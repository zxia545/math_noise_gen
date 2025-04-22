#!/usr/bin/env python3
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
# Ensure read_jsonl can handle potential errors per line gracefully
from utils import read_jsonl, write_jsonl, start_vllm_server, stop_vllm_server, chat_completion
# --- End assumed utilities ---

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
log_filename = f'validate_regenerate_running_{time.strftime("%d_%H_%M_%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    filename=log_filename,
    filemode='a', # Append mode
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Also log to console for interactive feedback
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# Avoid adding handler multiple times if script is re-run in same session
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)

logger.info(f"Starting the validation and regeneration script. Logging to {log_filename}")

# ------------------------------------------------------------------------------
# Helper Function to Find and Parse JSON containing a specific key - NEW
# ------------------------------------------------------------------------------
def find_and_parse_json_obj(s, key_literal):
    """
    Finds a JSON object string containing the specified key literal (e.g., '"my_key"')
    in string s, parses it, and returns the parsed object.
    Returns the parsed JSON object or None if not found or parsing fails.
    """
    if not isinstance(s, str) or not isinstance(key_literal, str):
        logger.debug(f"Invalid input type to find_and_parse_json_obj: s={type(s)}, key={type(key_literal)}")
        return None

    idx = s.find(key_literal)
    if idx == -1:
        logger.debug(f"Key literal '{key_literal}' not found in string.")
        return None

    # Find the opening brace '{' before the key
    start = s.rfind('{', 0, idx)
    if start == -1:
        logger.debug(f"Could not find opening brace '{{' before key '{key_literal}'.")
        return None

    # Scan forward to find the matching closing brace '}' using a stack
    stack_depth = 0
    for i in range(start, len(s)):
        char = s[i]
        if char == '{':
            stack_depth += 1
        elif char == '}':
            if stack_depth == 0:
                 # Found closing brace before opening one - malformed?
                 logger.debug(f"Found closing brace '}}' before matching opening brace, starting search from index {start}.")
                 return None
            stack_depth -= 1
            if stack_depth == 0:
                # Found the matching closing brace
                json_str = s[start:i+1]
                logger.debug(f"Found potential JSON string: {json_str[:100]}...")
                try:
                    # Attempt to parse the extracted string
                    parsed_obj = json.loads(json_str)
                    # Verify the key actually exists in the parsed object
                    # Need to remove quotes from key_literal for dict lookup
                    key_name = key_literal.strip('"')
                    if key_name in parsed_obj:
                        logger.debug(f"Successfully parsed JSON containing key '{key_name}'.")
                        return parsed_obj
                    else:
                        logger.warning(f"Parsed JSON object, but key '{key_name}' not found. Object: {parsed_obj}")
                        # Continue searching? For now, we assume the first found object containing the key literal text is the target.
                        return None # Treat as not found if key isn't in the final parsed object
                except json.JSONDecodeError as e:
                    logger.warning(f"Found JSON string boundaries but failed to parse: {e}\nString was: {json_str}")
                    return None # Parsing failed
    # If loop finishes without finding matching brace
    logger.debug(f"Could not find matching closing brace '}}' for opening brace at index {start}.")
    return None


# ------------------------------------------------------------------------------
# Prompt Constructors (Reused - No Changes Needed Here)
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
    question = question_dict.get('question', '')
    correct = question_dict.get("answer", '')
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
    parser = argparse.ArgumentParser(description="Validate and regenerate noise answers in a JSONL file with retries, using key-based JSON finding.")
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
        required=True, # Make model path required
        help="Path or name of the model for vLLM (used for both validation and regeneration)."
    )
    parser.add_argument("--model_name", type=str, default="validator_regenerator_model", help="Name of the model for vLLM.")
    parser.add_argument("--gpu", type=int, default=8, help="Number of GPUs for tensor parallel.")
    parser.add_argument("--port", type=int, default=8020, help="Port for the vLLM server.")
    parser.add_argument("--threads", type=int, default=30, help="Number of threads for concurrent processing.")
    parser.add_argument("--max_tokens_validate", type=int, default=512, help="Max tokens for the validation response.")
    parser.add_argument("--max_tokens_regenerate", type=int, default=2048, help="Max tokens for regenerated noise.")
    parser.add_argument("--temperature_validate", type=float, default=0.1, help="Temperature for validation (low for consistency).")
    parser.add_argument("--temperature_regenerate", type=float, default=0.7, help="Temperature for regeneration (higher for variability).")
    parser.add_argument("--save_interval", type=int, default=2000, help="Save progress every N records (default: 2000).")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for failed LLM calls (default: 3).")

    args = parser.parse_args()

    # --- Ensure output directory exists ---
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path, exist_ok=True)
        logger.info(f"Created output directory: {args.output_folder_path}")

    # --- Define output file path ---
    base_name = os.path.basename(args.input_jsonl)
    output_file = os.path.join(args.output_folder_path, f'validated_{base_name}')
    logger.info(f'Input file: {args.input_jsonl}')
    logger.info(f'Output file: {output_file}')

    # --- Check if output file already exists and handle resume ---
    processed_indices = set()
    if os.path.exists(output_file):
        logger.warning(f"Output file {output_file} already exists. Attempting to resume by checking processed indices.")
        try:
            for record in read_jsonl(output_file):
                 idx = record.get("idx")
                 if idx is not None:
                     processed_indices.add(idx)
            logger.info(f"Resuming. Found {len(processed_indices)} unique processed indices in existing output file.")
        except Exception as e:
            logger.error(f"Error reading existing output file for resuming: {e}. Indices might be incomplete.", exc_info=True)

    # --- Read input data ---
    try:
        input_data_list = list(read_jsonl(args.input_jsonl))
        logger.info(f"Loaded {len(input_data_list)} records from {args.input_jsonl}")
    except Exception as e:
        logger.critical(f"Failed to load input file {args.input_jsonl}: {e}", exc_info=True)
        sys.exit(1)

    # Filter out records that are already processed based on index
    records_to_process = [record for record in input_data_list if record.get("idx") not in processed_indices]
    if not records_to_process:
        logger.info("No new records to process (all indices found in existing output). Exiting.")
        sys.exit(0)
    else:
        logger.info(f"Filtered records. Will process {len(records_to_process)} new records (indices not found in output file).")


    # --- Start vLLM server ---
    logger.info("Starting vLLM server...")
    process = start_vllm_server(args.model, args.model_name, args.port, args.gpu)
    if not process:
         logger.critical("Failed to start vLLM server. Exiting.")
         sys.exit(1)

    # --- LLM Call Wrapper with Retry (using new JSON finder) ---
    def call_llm_with_retry(api_base, model_name, messages, max_tokens, temperature, target_json_key_literal):
        """
        Calls the LLM API with retry logic.
        Uses find_and_parse_json_obj to extract JSON containing target_json_key_literal.
        Returns the parsed JSON object and the raw response string on success, else (None, None).
        """
        raw_response = None # Keep track of the last raw response for regeneration payload
        for attempt in range(args.max_retries):
            try:
                raw_response = chat_completion( # Get the raw string response
                    api_base=api_base,
                    model_name=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                if not raw_response:
                    logger.warning(f"LLM call attempt {attempt + 1}/{args.max_retries} returned empty response.")
                    time.sleep(2 ** attempt) # Exponential backoff
                    continue

                # Use the new function to find and parse the JSON
                parsed_json = find_and_parse_json_obj(raw_response, target_json_key_literal)

                if parsed_json:
                    logger.debug(f"LLM call attempt {attempt + 1} successful. Found and parsed JSON containing key {target_json_key_literal}.")
                    return parsed_json, raw_response # Return parsed object and raw text
                else:
                    logger.warning(f"LLM call attempt {attempt + 1}/{args.max_retries}: Could not find/parse JSON with key {target_json_key_literal}. Raw response (start): {raw_response[:200]}...")

            except Exception as e:
                logger.error(f"LLM call attempt {attempt + 1}/{args.max_retries} failed with exception: {e}", exc_info=True)

            # Wait before retrying
            if attempt < args.max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        logger.error(f"LLM call failed after {args.max_retries} attempts for key {target_json_key_literal}.")
        # Return None for parsed, but potentially return the last raw response seen
        return None, raw_response


    # --- Processing Function ---
    def process_record(record):
        idx = record.get("idx", "N/A")
        question = record.get("question") or record.get("input")
        correct_answer = record.get("answer") or record.get("output")
        # original_noise_payload stores the *string* output from the first script,
        # which might contain JSON or just text.
        original_noise_payload_str = record.get("noise_answer_explanation")

        validation_result = {"is_valid_noise": False, "reason": "Validation not performed"}
        # This will store the string that goes into the final output file's
        # 'noise_answer_explanation' field. It should ideally be the raw string
        # containing the JSON object generated by the LLM.
        final_noise_payload_str = original_noise_payload_str
        noise_text_to_validate = None
        regenerated = False
        validation_status = "pending" # pending, success, failed_api, failed_parsing, skipped

        # --- Basic Check ---
        if not all([question, correct_answer]): # Allow missing noise payload initially
            logger.warning(f"Record {idx}: Missing question or answer. Skipping.")
            return {**record, "validation_status": "skipped_missing_fields", "noise_needs_regeneration": True} # Mark for regen? Or just skip?

        # --- 1. Extract Noise Text for Validation ---
        # Try to parse the JSON object containing 'noise_answer_with_steps' from the original payload string
        noise_key_literal = '"noise_answer_with_steps"' # Key expected in the original payload
        parsed_original_noise = find_and_parse_json_obj(original_noise_payload_str, noise_key_literal)

        if parsed_original_noise:
            noise_text_to_validate = parsed_original_noise.get(noise_key_literal.strip('"'))
            if noise_text_to_validate:
                 logger.debug(f"Record {idx}: Successfully extracted noise text for validation.")
            else:
                 logger.warning(f"Record {idx}: Parsed JSON from original payload, but '{noise_key_literal}' key had no value. Forcing regeneration.")
                 validation_result = {"is_valid_noise": False, "reason": f"Original noise payload JSON parsed but key '{noise_key_literal}' missing/empty."}
                 validation_status = "failed_parsing"
        else:
            # If we couldn't find/parse the specific JSON, maybe the payload is just raw text?
            # Or maybe it's empty/None. If it's a non-empty string, maybe try validating it?
            # For now, let's assume failure to find the JSON means regeneration is needed.
            logger.warning(f"Record {idx}: Could not find/parse JSON object with key {noise_key_literal} in original payload. Forcing regeneration.")
            validation_result = {"is_valid_noise": False, "reason": f"Original noise payload missing or invalid JSON structure with key {noise_key_literal}."}
            validation_status = "failed_parsing" # Or maybe "missing_payload" if original_noise_payload_str is None/empty

        # --- 2. Validate (if we have text to validate) ---
        if noise_text_to_validate:
            logger.info(f"Record {idx}: Validating noise...")
            validation_messages = construct_validation_message(question, correct_answer, noise_text_to_validate)
            validation_key_literal = '"is_valid_noise"' # Key expected from validation LLM
            validation_response_json, _ = call_llm_with_retry(
                api_base=f"http://localhost:{args.port}/v1",
                model_name=args.model_name,
                messages=validation_messages,
                max_tokens=args.max_tokens_validate,
                temperature=args.temperature_validate,
                target_json_key_literal=validation_key_literal
            )

            # Check the parsed JSON result
            if validation_response_json and isinstance(validation_response_json.get(validation_key_literal.strip('"')), bool):
                validation_result = validation_response_json # Contains 'is_valid_noise' and 'reason'
                validation_status = "success"
                logger.info(f"Record {idx}: Validation result: {validation_result.get('is_valid_noise')}. Reason: {validation_result.get('reason')}")
            else:
                logger.error(f"Record {idx}: Validation failed after retries (API error or invalid/incomplete JSON response).")
                validation_result = {"is_valid_noise": False, "reason": "Validation LLM call failed or gave malformed/incomplete JSON response after retries."}
                validation_status = "failed_api"

        # --- 3. Regenerate if Necessary ---
        if not validation_result.get("is_valid_noise"):
            logger.warning(f"Record {idx}: Noise invalid or validation failed. Attempting regeneration (Reason: {validation_result.get('reason', 'N/A')}).")
            regenerated = True
            gen_question_dict = {"question": question, "answer": correct_answer}
            regeneration_messages = construct_noise_gen_message(gen_question_dict)
            regeneration_key_literal = '"noise_answer_with_steps"' # Key expected from regeneration LLM

            # We need the raw response string from the regeneration call to save in the output file
            new_noise_json, new_noise_payload_str_raw = call_llm_with_retry(
                api_base=f"http://localhost:{args.port}/v1",
                model_name=args.model_name,
                messages=regeneration_messages,
                max_tokens=args.max_tokens_regenerate,
                temperature=args.temperature_regenerate,
                target_json_key_literal=regeneration_key_literal
            )

            if new_noise_json and new_noise_payload_str_raw:
                # Success! Update the payload string to be saved.
                final_noise_payload_str = new_noise_payload_str_raw
                logger.info(f"Record {idx}: Successfully regenerated noise.")
                # Optionally re-validate here if desired
            else:
                logger.error(f"Record {idx}: Regeneration failed after {args.max_retries} attempts. Keeping original/invalid noise payload string.")
                validation_result["reason"] += " | Regeneration attempt failed after retries."
                regenerated = False # Mark as failed
                # final_noise_payload_str remains the original invalid one

        # --- 4. Construct Final Output Record ---
        output_record = {
            **record, # Keep all original fields
            # Save the string payload (either original or the raw response from regen)
            "noise_answer_explanation": final_noise_payload_str,
            "validation_passed": validation_result.get("is_valid_noise", False),
            "validation_reason": validation_result.get("reason", "Unknown"),
            "noise_was_regenerated": regenerated,
            "validation_run_status": validation_status
        }
        return output_record

    # --- Process data with ThreadPoolExecutor ---
    processed_count = 0
    last_save_count = 0
    start_time = time.time()
    results_buffer = [] # Hold results before saving

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_record, record): record.get("idx") for record in records_to_process}

        for future in futures:
            idx = futures[future]
            try:
                result = future.result()
                results_buffer.append(result)
                processed_count += 1

                if processed_count % 100 == 0:
                     current_time = time.time()
                     elapsed_time = current_time - start_time
                     rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                     logger.info(f"Progress: Processed {processed_count}/{len(records_to_process)} records ({len(processed_indices)} skipped). Rate: {rate:.2f} rec/s.")

                if processed_count > 0 and processed_count % args.save_interval == 0:
                    logger.warning(f"Saving intermediate results ({len(results_buffer)} new records)...")
                    write_jsonl(output_file, results_buffer, append=True)
                    results_buffer = []
                    last_save_count = processed_count
                    logger.info(f"Intermediate save complete. Total processed this run: {processed_count}")

            except Exception as e:
                logger.critical(f"Record {idx}: CRITICAL error processing future: {e}", exc_info=True)
                results_buffer.append({
                    "idx": idx,
                    "error": f"Critical failure during processing: {str(e)}",
                    "validation_run_status": "critical_error"
                 })
                processed_count += 1 # Ensure count increments

    # --- Save any remaining records ---
    if results_buffer:
        logger.info(f"Saving final batch of {len(results_buffer)} records...")
        write_jsonl(output_file, results_buffer, append=True)
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