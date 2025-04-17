# utils.py
import os
import time
import json
import requests
from typing import Dict, Any, List
import subprocess
from openai import OpenAI
import json
import os
import codecs


def filter_and_fix_file(file_path):
    """
    Reads a JSONL file, removes invalid lines, and overwrites the original file with only valid lines.
    """
    valid_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.strip():  # Check if the line is not empty
                try:
                    json.loads(line)  # Attempt to load the line as JSON
                    valid_lines.append(line)  # Store valid lines
                except json.JSONDecodeError:
                    print(f"Invalid JSON line removed: {line.strip()}")  # Log invalid line
    
    # Overwrite the original file with valid lines
    with open(file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(valid_lines)

def read_jsonl(file_path):
    """
    Reads a JSONL file, ensuring proper UTF-8 handling and fixing any Unicode escape sequences.
    Yields each JSON object as a dictionary.
    """
    filter_and_fix_file(file_path)  # Ensure invalid lines are removed
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)  # Load JSON and decode Unicode properly
                    yield data
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Skipping invalid JSON line in {file_path}: {line} - {e}")


def write_jsonl(file_path, data_list, append=False):
    """
    Writes a list of dictionaries to a JSONL file with proper UTF-8 encoding.
    Ensures Unicode characters are stored correctly without escaping.
    """
    mode = 'a' if append else 'w'

    # Ensure the directory exists before writing
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with codecs.open(file_path, mode, encoding='utf-8') as f:
        for item in data_list:
            # Decode any escaped Unicode characters before writing
            json_line = json.dumps(item, ensure_ascii=False)
            json_line = json_line.encode('utf-8').decode('utf-8')  # Ensures proper encoding
            
            f.write(json_line + '\n')
            

def chat_completion(api_base: str, model_name: str, messages: list, max_tokens=256, temperature=0.7):
    """
    Generic helper that uses the new openai client interface to get a chat completion.
    """
    
    if '/v1' not in api_base:
        api_base = api_base + '/v1'
    
    client = OpenAI(base_url=api_base, api_key="xxx")  # point to the local vLLM server
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return completion.choices[0].message.content



def start_vllm_server(model_path: str, model_name: str, port: int, gpu: int = 1):
    """
    Launches a vLLM OpenAI API server via subprocess.
    model_path: The path or name of the model you want to host
    port: Which port to host on
    gpu: The tensor-parallel-size (number of GPUs)
    """
    # Command to activate conda environment and start the server
    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={model_path}',
        f"--served-model-name={model_name}",
        f'--tensor-parallel-size={gpu}',
        f"--gpu-memory-utilization=0.82",
        f'--port={port}',
        '--trust-remote-code'
    ]

    process = subprocess.Popen(command, shell=False)
    
    wait_for_server(f"http://localhost:{port}", 1500)
    
    print(f"[INFO] Started vLLM server for model '{model_path}' on port {port} (GPU={gpu}).")

    return process


def start_vllm_server_with_gpus(model_path: str, model_name: str, port: int, gpus: List[int]):
    """
    Launches a vLLM OpenAI API server via subprocess with specific GPUs assigned.

    Parameters:
    model_path: str - The path or name of the model you want to host.
    model_name: str - The name of the model to be served.
    port: int - The port to host the server on.
    gpus: List[int] - List of GPU indices to be assigned for this server.

    Returns:
    process: subprocess.Popen - The process running the vLLM server.
    """
    gpu_list = ",".join(map(str, gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={model_path}',
        f'--served-model-name={model_name}',
        f'--tensor-parallel-size={len(gpus)}',
        '--gpu-memory-utilization=0.85',
        f'--port={port}',
        '--trust-remote-code'
    ]

    process = subprocess.Popen(command, shell=False, env=os.environ.copy())
    
    wait_for_server(f"http://localhost:{port}", 600)

    print(f"[INFO] Started vLLM server for model '{model_name}' on port {port} with GPUs {gpu_list}.")

    return process

def allocate_gpus(total_gpus: int, processes: int) -> List[List[int]]:
    """
    Allocate GPUs for multiple processes.

    Parameters:
    total_gpus: int - Total number of GPUs available.
    processes: int - Number of processes to allocate GPUs for.

    Returns:
    List[List[int]] - A list where each sublist contains the GPUs assigned to a process.
    """
    if total_gpus < processes:
        raise ValueError("Not enough GPUs available for the number of processes.")

    gpus_per_process = total_gpus // processes
    extra_gpus = total_gpus % processes

    allocation = []
    start = 0

    for i in range(processes):
        end = start + gpus_per_process + (1 if i < extra_gpus else 0)
        allocation.append(list(range(start, end)))
        start = end

    return allocation



def wait_for_server(url: str, timeout: int = 600):
    """
    Polls the server's /models endpoint until it responds with HTTP 200 or times out.
    """
    start_time = time.time()
    while True:
        try:
            r = requests.get(url + "/v1/models", timeout=3)
            if r.status_code == 200:
                print("[INFO] vLLM server is up and running.")
                return
        except Exception:
            pass
        if time.time() - start_time > timeout:
            raise RuntimeError(f"[ERROR] Server did not start at {url} within {timeout} seconds.")
        time.sleep(2)
        
def stop_vllm_server(process):
    process.terminate()
    process.wait()
    print("[INFO] Stopped vLLM server.")



def create_output_directory(model_name: str):
    """
    Creates the output directory named after the LLM model, if it doesn't exist.
    Returns the path to that directory.
    """
    output_dir = os.path.join("outputs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
