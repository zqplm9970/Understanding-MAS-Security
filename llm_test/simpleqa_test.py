import os
import json
import time
from datetime import datetime
from typing import Dict, Tuple

import chardet
import pandas as pd
import re
import ast
from collections import Counter

from openai import OpenAI

API_KEY = "your key"
BASE_URL = "your url"
MODEL_NAME = "your model"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

csv_path = "data/SimpleQA/SimpleQA_sample_433.csv"
RESULTS_DIR = "llm_test/results"
RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "SingleModel_Qwen2.5-7B-Instruct_SimpleQA.json"
)

START_QUESTION_IDX = -1
EVAL_MODE = "SingleModel_Intrinsic_Performance"


def load_csv_data(csv_path: str) -> pd.DataFrame:
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding'] or 'utf-8'
    df = pd.read_csv(csv_path, encoding=encoding)

    expected_cols = {"metadata", "problem", "answer"}
    missing = expected_cols - set(df.columns)
    return df


def safe_literal_eval(s: str):
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        d = {}
        m_topic = re.search(r"topic'\s*:\s*'([^']+)'", s)
        if m_topic:
            d['topic'] = m_topic.group(1)
        m_type = re.search(r"answer_type'\s*:\s*'([^']+)'", s)
        if m_type:
            d['answer_type'] = m_type.group(1)
        return d


def extract_metadata_info(metadata_field) -> Tuple[str, str]:
    md = safe_literal_eval(metadata_field)
    topic = md.get('topic', '')
    answer_type = md.get('answer_type', '')
    return (
        str(topic).strip() if topic else "Other",
        str(answer_type).strip() if answer_type else "Other"
    )


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.strip('"').strip("'")
    s = s.rstrip(" .,:;!?")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def build_simpleqa_prompt(problem: str, topic: str, answer_type: str) -> str:
    return (
        f"Topic: {topic}\n"
        f"Answer Type: {answer_type}\n"
        f"Question: {problem}\n\n"
        f"Please directly provide the final answer.\n"
        f"Output format: output ONLY the answer itself. "
        f"Do NOT include explanations or any additional text."
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
            "dataset": "SimpleQA_sample_433",
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


def evaluate_correctness(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


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
                        "You are an assistant for answering question-answering tasks. "
                        "The user will provide a SimpleQA question along with its topic "
                        "and answer type. "
                        "Your task is to provide the most appropriate final answer.\n"
                        "IMPORTANT: Output ONLY the answer itself. "
                        "Do NOT include explanations, prefixes, or any additional text."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Model call error: {e}")
        return f"Error: {e}"


def run_single_question(
    full_prompt: str,
    question_idx: int,
    ground_truth: str,
    topic: str,
    answer_type: str,
    problem: str,
    metadata_raw
) -> Dict:

    model_answer = call_model(full_prompt)

    is_correct = evaluate_correctness(model_answer or "", ground_truth or "")

    question_result = {
        "question_idx": question_idx,
        "problem": problem,
        "topic": topic,
        "answer_type": answer_type,
        "prompt": full_prompt,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "is_correct": is_correct,
        "metadata_raw": str(metadata_raw),
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

        metadata_raw = item['metadata']
        problem = str(item['problem'])
        ground_truth = str(item['answer']).strip()

        topic, answer_type = extract_metadata_info(metadata_raw)
        full_prompt = build_simpleqa_prompt(problem, topic, answer_type)

        try:
            question_result = run_single_question(
                full_prompt,
                idx,
                ground_truth,
                topic,
                answer_type,
                problem,
                metadata_raw
            )

            processed_count += 1
            if question_result.get("is_correct", False):
                correct_count += 1

            all_results["results"].append(question_result)
            save_all_results(all_results, RESULTS_FILE)

        except Exception as e:
            error_result = {
                "question_idx": idx,
                "problem": problem,
                "topic": topic,
                "answer_type": answer_type,
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
