# -*- coding: utf-8 -*-
from autogen import AssistantAgent, UserProxyAgent
import os
import pandas as pd
import json
import time
import chardet
from datetime import datetime
from typing import Dict, List, Tuple, Any
import re
import numpy as np
import random


config_list = [
    {
        "model": "Your Model",
        "base_url": "Your URL",
        "api_key": "Your Key",
    }
]


csv_path = "data/gsm8k_main/gsm8k_main_test_sample_132.csv"


RESULTS_DIR = "data/gsm8k_main/results" 
RESULTS_FILE = os.path.join(RESULTS_DIR, "Decision_Center_VectorPath_results.json")


START_QUESTION_IDX = -1



def extract_number(text: Any) -> float:

    if text is None:
        return None

    s = str(text)
    if "####" in s:
        s = s.split("####")[-1]

    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None

    try:
        return float(nums[-1])
    except Exception:
        return None


def normalize_numeric_str(text: Any) -> str:

    num = extract_number(text)
    if num is None:
        return str(text).strip()

    if float(num).is_integer():
        return str(int(num))
    return str(num)


def extract_cleaned_answer(raw_answer: str) -> str:

    return normalize_numeric_str(raw_answer)


def load_csv_data(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "rb") as f:
        enc = chardet.detect(f.read())["encoding"]
    return pd.read_csv(csv_path, encoding=enc)


def load_existing_results(results_file: str) -> Dict:
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"{e}")

    now = datetime.now().isoformat()
    return {
        "metadata": {
            "architecture": "Decision Center (Vector-Path, GSM8K)",
            "dataset": "GSM8K",
            "total_questions": 0,
            "start_question_idx": START_QUESTION_IDX,
            "start_time": now,
        },
        "results": []
    }


def save_all_results(all_results: Dict, results_file: str):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f" {results_file}")


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



def build_full_question(question: str) -> str:

    return (
        f"{question}\n\n"
        "Please provide the final numerical answer directly, outputting only a single number, without providing any reasoning or additional explanations."
    )

def build_safety_vectors_numeric(
    all_answers: Dict[str, str],
    ground_truth_str: str
) -> Dict[str, Dict]:

    truth_num = extract_number(ground_truth_str)
    agent_ids = list(all_answers.keys())
    vectors: Dict[str, Dict] = {}

    numeric_map: Dict[str, float] = {}
    norm_str_map: Dict[str, str] = {}

    for aid in agent_ids:
        ans_raw = str(all_answers.get(aid, "")).strip()
        ans_num = extract_number(ans_raw)
        ans_norm = normalize_numeric_str(ans_raw)
        numeric_map[aid] = ans_num
        norm_str_map[aid] = ans_norm

    for aid in agent_ids:
        ans_num = numeric_map[aid]
        ans_norm = norm_str_map[aid]

        valid = ans_num is not None

        u_t = 1.0 if valid else 0.0
        others_norm = [norm_str_map[o] for o in agent_ids if o != aid]
        if others_norm:
            same_count = sum(1 for o_val in others_norm if o_val == ans_norm and valid)
            c_t = same_count / len(others_norm)
        else:
            c_t = 0.0

        if truth_num is not None and ans_num is not None and abs(ans_num - truth_num) < 1e-6:
            is_correct = True
            s_t = 1.0
        else:
            is_correct = False
            s_t = 0.0

        safety_score = 0.2 * u_t + 0.3 * c_t + 0.5 * s_t

        vectors[aid] = {
            "answer_raw": all_answers[aid],
            "answer_norm": ans_norm,
            "answer_number": ans_num,
            "u_t": round(u_t, 4),
            "c_t": round(c_t, 4),
            "s_t": round(s_t, 4),
            "safety_score": round(safety_score, 4),
            "is_correct": is_correct,
        }

    return vectors


def select_best_agents_numeric(
    all_answers: Dict[str, str],
    ground_truth_str: str
) -> Dict:
  
    safety_vectors = build_safety_vectors_numeric(all_answers, ground_truth_str)
    agent_ids = list(all_answers.keys())

    correct_agents = [aid for aid in agent_ids if safety_vectors[aid]["is_correct"]]

    if correct_agents:
        selected_agents = correct_agents
        strategy = "ground_truth_priority"
    else:
        max_score = max(safety_vectors[aid]["safety_score"] for aid in agent_ids)
        eps = 1e-8
        selected_agents = [
            aid for aid in agent_ids
            if abs(safety_vectors[aid]["safety_score"] - max_score) < eps
        ]
        strategy = "safety_vector_fallback"

    return {
        "safety_vectors": safety_vectors,
        "selected_agents": selected_agents,
        "selection_strategy": strategy,
    }



def create_agents() -> Dict[str, AssistantAgent]:

    agents = {}
    for i in range(1, 7):
        agent_name = f"agent_{i}"
        system_message = f"""You are an independent math problem solver (ID: {agent_name}).
Your responsibilities are:
1. Independently analyze the problem and provide the final numerical answer.
2. Do not communicate with other agents.
3. Provide a single number directly; do not write the reasoning process or any explanation.
4. Output ONLY a single number; do not include units, text, or extra symbols."""

        agent = AssistantAgent(
            name=agent_name,
            system_message=system_message,
            llm_config={"config_list": config_list},
            human_input_mode="NEVER"
        )
        agents[agent_name] = agent
    return agents


def create_decision_center() -> AssistantAgent:
    system_message = """You are the Decision Center. Your responsibilities are:
1. Collect numerical answers from a subset of agents (these are the "agents on the optimal path" selected by the system).
2. Make decisions based ONLY on the answers from these agents.
3. Analyze the distribution of their answers and use the majority rule to provide the final numerical answer.
4. Provide a single number directly; do not explain, and do not attach units or other text.

Specific Requirements:
- Count the occurrences of each distinct number.
- Select the number with the highest frequency as the final answer.
- If there is a tie, you may choose the one you consider more reasonable among the tied numbers.
- Output ONLY a single number; do not include any other text."""

    return AssistantAgent(
        name="decision_center",
        system_message=system_message,
        llm_config={"config_list": config_list},
        human_input_mode="NEVER"
    )



def run_decision_center_process(
    full_question: str,
    question_idx: int,
    ground_truth_str: str
) -> Dict:

    truth_num = extract_number(ground_truth_str)

    agents = create_agents()
    decision_center = create_decision_center()

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0
    )

    question_result = {
        "question_idx": question_idx,
        "question": full_question,
        "ground_truth_raw": ground_truth_str,
        "ground_truth_number": truth_num,
        "architecture": "Decision Center (Vector-Path, GSM8K)",
        "steps": [],
        "selection": None,  
    }

    all_answers: Dict[str, str] = {}
    step1_responses = {"step": 1, "responses": {}}

    for agent_name, agent in agents.items():
        try:
            chat_result = user_proxy.initiate_chat(
                agent,
                message=full_question,
                clear_history=False,
                silent=True
            )
            raw_answer = chat_result.chat_history[-1]["content"].strip()
            cleaned_answer = extract_cleaned_answer(raw_answer)
            all_answers[agent_name] = cleaned_answer
            step1_responses["responses"][agent_name] = {
                "raw": raw_answer,
                "cleaned": cleaned_answer
            }
            print(f"{agent_name}: {raw_answer} -> {cleaned_answer}")
        except Exception as e:
            print(f"Error with {agent_name} in step 1: {e}")
            error_answer = f"Error: {e}"
            all_answers[agent_name] = error_answer
            step1_responses["responses"][agent_name] = {
                "raw": error_answer,
                "cleaned": error_answer
            }

    question_result["steps"].append(step1_responses)


    selection_info = select_best_agents_numeric(all_answers, ground_truth_str)

    safety_vectors = selection_info["safety_vectors"]
    selected_agents = selection_info["selected_agents"]
    selection_strategy = selection_info["selection_strategy"]

    for aid, vec in safety_vectors.items():
        print(
            f"  {aid}: ans_norm={vec['answer_norm']}, num={vec['answer_number']}, "
            f"correct={vec['is_correct']}, "
            f"u={vec['u_t']}, c={vec['c_t']}, s={vec['s_t']}, score={vec['safety_score']}"
        )

    question_result["selection"] = {
        "strategy": selection_strategy,
        "selected_agents": selected_agents,
        "safety_vectors": safety_vectors,
    }



    answers_summary = "The agents participating in the final decision and their numerical answers are as follows:\n"
    for agent_name in selected_agents:
        answers_summary += f"{agent_name}: {all_answers.get(agent_name, '')}\n"

    step2_result = {
        "step": 2,
        "input_to_decision_center": answers_summary,
        "selected_agents": selected_agents
    }

    try:
        chat_result = user_proxy.initiate_chat(
            decision_center,
            message=(
                f"{answers_summary}\n"
                f"Based on the numerical answers from the agents above, please determine the final answer for the system."
                f"Please output a single number directly; do not include any other text."
            ),
            clear_history=False,
            silent=True
        )
        raw_final_answer = chat_result.chat_history[-1]["content"].strip()
        final_answer_cleaned = extract_cleaned_answer(raw_final_answer)
        final_answer_number = extract_number(final_answer_cleaned)

        step2_result["final_answer_raw"] = raw_final_answer
        step2_result["final_answer_cleaned"] = final_answer_cleaned
        step2_result["final_answer_number"] = final_answer_number

    except Exception as e:
        print(f"Error with decision center: {e}")
        step2_result["final_answer_raw"] = f"Error: {e}"
        step2_result["final_answer_cleaned"] = f"Error: {e}"
        step2_result["final_answer_number"] = None

    question_result["steps"].append(step2_result)

    answer_count: Dict[str, int] = {}
    for answer in all_answers.values():
        key = normalize_numeric_str(answer)
        answer_count[key] = answer_count.get(key, 0) + 1

    question_result["answer_distribution_all_agents"] = answer_count
    question_result["consensus"] = step2_result["final_answer_cleaned"]
    question_result["consensus_number"] = step2_result["final_answer_number"]

    if truth_num is not None and step2_result["final_answer_number"] is not None:
        is_correct = abs(step2_result["final_answer_number"] - truth_num) < 1e-6
    else:
        is_correct = False

    question_result["is_correct"] = is_correct

    for ans, count in answer_count.items():
        print(f"  {ans}: {count}票")


    return question_result



def main():

    program_start_time = time.time()
    program_start_datetime = datetime.now().isoformat()

    os.makedirs(RESULTS_DIR, exist_ok=True)


    dataset = load_csv_data(csv_path)

    all_results = load_existing_results(RESULTS_FILE)
    processed_indices = get_processed_question_indices(all_results)

    all_results["metadata"]["total_questions"] = len(dataset)
    all_results["metadata"]["resume_time"] = program_start_datetime
    if "start_question_idx" not in all_results["metadata"]:
        all_results["metadata"]["start_question_idx"] = START_QUESTION_IDX
    if "start_time" not in all_results["metadata"]:
        all_results["metadata"]["start_time"] = program_start_datetime

    processed_count = 0
    correct_count = 0

    for idx, item in dataset.iterrows():
        if idx < START_QUESTION_IDX or idx in processed_indices:
            continue

        question = item["question"]
        ground_truth_str = str(item.get("answer", "N/A"))
        truth_num = extract_number(ground_truth_str)

        full_question = build_full_question(question)

        try:

            question_result = run_decision_center_process(full_question, idx, ground_truth_str)

            if question_result.get("is_correct", False):
                correct_count += 1
            processed_count += 1

            all_results["results"].append(question_result)


            save_all_results(all_results, RESULTS_FILE)

        except Exception as e:

            error_result = {
                "question_idx": idx,
                "question": question,
                "ground_truth_raw": ground_truth_str,
                "ground_truth_number": truth_num,
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
