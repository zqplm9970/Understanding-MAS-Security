import os
import json
import time
from datetime import datetime
from typing import Dict, List

import chardet
import pandas as pd
import numpy as np
import re

from openai import OpenAI


API_KEY = "your key"
BASE_URL = "your url"
MODEL_NAME = "your model"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

csv_path = "data/commonsense_qa/commonsense_qa.csv"
RESULTS_DIR = "llm_test/results"
RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "SingleModel_Qwen2.5-7B-Instruct_CommonsenseQA.json"
)

START_QUESTION_IDX = -1
EVAL_MODE = "SingleModel_Intrinsic_Performance_CommonsenseQA"


def parse_choices(choices_str: str) -> List[str]:
    try:
        choices_str = choices_str.replace("array", "np.array")
        choices_dict = eval(choices_str, {"np": np})

        labels = choices_dict.get('label', [])
        texts = choices_dict.get('text', [])

        if len(labels) == len(texts):
            options = [f"{labels[i]}.{texts[i]}" for i in range(len(labels))]
            return options
        else:
            return []
    except Exception as e:
        print(f"{e}")
        return []


def load_csv_data(csv_path: str) -> pd.DataFrame:
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding'] or 'utf-8'
    df = pd.read_csv(csv_path, encoding=encoding)

    expected_cols = {"question", "choices", "answerKey"}
    missing = expected_cols - set(df.columns)
    return df


def build_question_with_options(question: str, options: List[str]) -> str:
    options_text = "\n".join(options)
    return (
        "Below is a commonsense multiple-choice question. "
        "Please select the most appropriate answer from the given options.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Please output ONLY the option letter you choose (A, B, C, D, or E). "
        "Do NOT output the option content or any explanation."
    )


def load_existing_results(results_file: str) -> Dict:
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"{e}")

    return {
        "metadata": {
            "eval_mode": EVAL_MODE,
            "model": MODEL_NAME,
            "dataset": "CommonsenseQA",
            "total_questions": 0,
            "start_question_idx": START_QUESTION_IDX,
            "start_time": datetime.now().isoformat()
        },
        "results": []
    }


def save_all_results(all_results: Dict, results_file: str):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)


def extract_answer_from_response(response: str) -> str:
    response = response.strip().upper()
    matches = re.findall(r'[A-E]', response)
    if matches:
        return matches[0]
    return response


def evaluate_correctness(pred: str, gold: str) -> bool:
    if not pred:
        return False
    return pred.strip().upper() == str(gold).strip().upper()


def get_processed_question_indices(all_results: Dict) -> set:
    processed_indices = set()
    for result in all_results["results"]:
        if "question_idx" in result:
            processed_indices.add(result["question_idx"])
    return processed_indices


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}"


def call_model(full_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a commonsense multiple-choice question answering assistant. "
                        "The user will provide an English question and several options (A–E). "
                        "Your task is to select the most appropriate option.\n"
                        "IMPORTANT: Output ONLY the option letter (A, B, C, D, or E). "
                        "Do NOT output any explanation or additional content."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{e}")
        return f"Error: {e}"


def run_single_question(
    question: str,
    options: List[str],
    question_idx: int,
    ground_truth: str,
) -> Dict:
    full_prompt = build_question_with_options(question, options)
    raw_answer = call_model(full_prompt)
    pred = extract_answer_from_response(raw_answer)

    is_correct = evaluate_correctness(pred, ground_truth)

    question_result = {
        "question_idx": question_idx,
        "question": question,
        "options": options,
        "prompt": full_prompt,
        "ground_truth": ground_truth,
        "model_raw_answer": raw_answer,
        "model_parsed_answer": pred,
        "is_correct": is_correct,
        "timestamp": datetime.now().isoformat()
    }
    return question_result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    program_start_time = time.time()
    program_start_datetime = datetime.now().isoformat()

    dataset = load_csv_data(csv_path)

    all_results = load_existing_results(RESULTS_FILE)
    processed_indices = get_processed_question_indices(all_results)

    if "start_question_idx" not in all_results["metadata"]:
        all_results["metadata"]["start_question_idx"] = START_QUESTION_IDX

    all_results["metadata"]["total_questions"] = len(dataset)

    if "start_time" not in all_results["metadata"]:
        all_results["metadata"]["start_time"] = program_start_datetime

    all_results["metadata"]["resume_time"] = program_start_datetime

    correct_count = 0
    processed_count = 0

    for idx, item in dataset.iterrows():
        if idx < START_QUESTION_IDX or idx in processed_indices:
            continue

        question = str(item["question"])
        choices_str = str(item["choices"])
        ground_truth = str(item["answerKey"]).strip().upper()

        options = parse_choices(choices_str)

        try:
            question_result = run_single_question(
                question,
                options,
                idx,
                ground_truth,
            )

            processed_count += 1
            if question_result.get("is_correct", False):
                correct_count += 1

            all_results["results"].append(question_result)
            save_all_results(all_results, RESULTS_FILE)

        except Exception as e:
            error_result = {
                "question_idx": idx,
                "question": question,
                "options": options,
                "ground_truth": ground_truth,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            all_results["results"].append(error_result)
            save_all_results(all_results, RESULTS_FILE)

    program_end_time = time.time()
    total_runtime_seconds = program_end_time - program_start_time
    program_end_datetime = datetime.now().isoformat()

    all_results["metadata"]["end_time"] = program_end_datetime
    all_results["metadata"]["processed_count"] = processed_count
    all_results["metadata"]["correct_count"] = correct_count
    all_results["metadata"]["accuracy"] = (
        round(correct_count / processed_count, 4) if processed_count > 0 else 0.0
    )
    all_results["metadata"]["total_runtime_seconds"] = round(total_runtime_seconds, 2)
    all_results["metadata"]["total_runtime_formatted"] = format_duration(total_runtime_seconds)
    all_results["metadata"]["average_time_per_question"] = (
        round(total_runtime_seconds / processed_count, 2) if processed_count > 0 else 0
    )

    save_all_results(all_results, RESULTS_FILE)


if __name__ == "__main__":
    main()
