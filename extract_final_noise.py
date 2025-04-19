#!/usr/bin/env python3
import argparse
import json
import re
import os
import sys

def find_json_obj(s, key):
    """
    Find a JSON object containing the specified key in string s.
    Returns the JSON substring or None if not found.
    """
    idx = s.find(key)
    if idx == -1:
        return None
    # Find the opening brace before the key
    start = s.rfind('{', 0, idx)
    if start == -1:
        return None
    # Scan forward to find the matching closing brace
    stack = []
    for i in range(start, len(s)):
        if s[i] == '{':
            stack.append('{')
        elif s[i] == '}':
            stack.pop()
            if not stack:
                return s[start:i+1]
    return None

def main():
    parser = argparse.ArgumentParser(description="Extract noise_answer_with_steps from a JSONL file")
    parser.add_argument('input', help='Path to the input JSONL file')
    parser.add_argument('-o', '--output', help='Path for the output JSONL (successes only)', default=None)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or os.path.splitext(input_path)[0] + '_extracted.jsonl'

    total = 0
    success = 0
    extracted_records = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            expl = record.get('noise_answer_explanation', '')
            json_str = find_json_obj(expl, 'noise_answer_with_steps')
            if json_str:
                try:
                    data = json.loads(json_str)
                    record['noise_answer_explanation'] = data
                    extracted_records.append(record)
                    success += 1
                except json.JSONDecodeError:
                    # Found substring but failed to parse
                    pass

    # Write only successfully extracted records
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for rec in extracted_records:
            outfile.write(json.dumps(rec, ensure_ascii=False) + '\n')

    rate = success / total if total else 0
    print(f"Processed {total} records.")
    print(f"Successfully extracted: {success}")
    print(f"Success rate: {rate:.2%}")

if __name__ == '__main__':
    main()
