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
import ast
import random

config_list = [
    {
        "model": "Your Model",
        "base_url": "Your URL",
        "api_key": "Your Key",
    }
]

csv_path = "data/SimpleQA/SimpleQA_sample_433.csv"
RESULTS_DIR = "data/SimpleQA/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "Decision_Center_VectorPath_results.json")
START_QUESTION_IDX = -1

def load_csv_data(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "rb") as f:
        result = chardet.detect(f.read())
        encoding = result["encoding"] or "utf-8"
    df = pd.read_csv(csv_path, encoding=encoding)
    expected_cols = {"metadata", "problem", "answer"}
    missing = expected_cols - set(df.columns)
    return df

def safe_literal_eval(s: Any):
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
    return (str(topic).strip() if topic else "Other",
            str(answer_type).strip() if answer_type else "Other")

def build_simpleqa_prompt(problem: str, topic: str, answer_type: str) -> str:
    return (
        f"Topic: {topic}\n"
        f"Answer Type: {answer_type}\n"
        f"Problem: {problem}\n\n"
        f"Please provide the final answer.\n"
        f"Output Requirement: Output ONLY the answer itself; do not explain or include extra text."
    )

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
            "architecture": "Decision Center (Vector-Path)",
            "dataset": "SimpleQA_sample_433",
            "total_questions": 0,
            "start_question_idx": START_QUESTION_IDX,
            "start_time": now,
            "resume_time": now,
            "end_time": None,
            "processed_count": 0,
            "correct_count": 0,
            "accuracy": 0.0,
            "total_runtime_seconds": 0.0,
            "total_runtime_formatted": "",
            "average_time_per_question": 0.0,
        },
        "results": []
    }

def save_all_results(all_results: Dict, results_file: str):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"All results saved to: {results_file}")

def get_processed_question_indices(all_results: Dict) -> set:
    processed_indices = set()
    for result in all_results["results"]:
        if "question_idx" in result:
            processed_indices.add(result["question_idx"])
    return processed_indices

def extract_answer_from_response(response: str) -> str:
    return str(response).strip()

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def normalize_answer_text(text: Any) -> str:
    s = str(text)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def build_safety_vectors(all_answers: Dict[str, str], ground_truth: str) -> Dict[str, Dict]:
    ground_truth_norm = normalize_answer_text(ground_truth)
    agent_ids = list(all_answers.keys())
    vectors: Dict[str, Dict] = {}
    for aid in agent_ids:
        ans_raw = str(all_answers.get(aid, ""))
        ans_norm = normalize_answer_text(ans_raw)
        valid = (ans_norm != "")
        is_correct = valid and (ground_truth_norm != "") and (ans_norm == ground_truth_norm)
        u_t = 1.0 if valid else 0.0
        others_norm = [
            normalize_answer_text(all_answers[o]) for o in agent_ids if o != aid
        ]
        if others_norm:
            same_count = sum(1 for o_ans in others_norm if o_ans == ans_norm and ans_norm != "")
            c_t = same_count / len(others_norm)
        else:
            c_t = 0.0
        s_t = 1.0 if is_correct else 0.0
        safety_score = 0.2 * u_t + 0.3 * c_t + 0.5 * s_t
        vectors[aid] = {
            "answer": ans_raw,
            "normalized_answer": ans_norm,
            "valid": valid,
            "is_correct": is_correct,
            "u_t": round(u_t, 4),
            "c_t": round(c_t, 4),
            "s_t": round(s_t, 4),
            "safety_score": round(safety_score, 4),
        }
    return vectors

def select_best_agents(all_answers: Dict[str, str], ground_truth: str) -> Dict:
    safety_vectors = build_safety_vectors(all_answers, ground_truth)
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
        system_message = f"""You are an independent reasoner. Your responsibilities are:
1. Independently analyze the problem and provide an answer.
2. Do not communicate with other agents.
3. Provide the answer directly; do not explain.
4. Answer format: Output ONLY the final answer itself; do not output any explanation or extra text.

Please carefully read the problem, combine the topic and answer type, and provide the most appropriate answer."""
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
1. Collect answers from a subset of agents (these are the "agents on the optimal path" selected by the system).
2. Make decisions based ONLY on the answers from these agents.
3. Analyze the distribution of their answers and use the majority rule to provide the final answer.
4. Provide the final answer directly; do not explain.

For open-ended Q&A, please analyze the answer text from these agents and select the most frequent or semantically consistent answer.
Output Requirement: Output ONLY the final answer itself; do not output any explanation or additional text."""
    return AssistantAgent(
        name="decision_center",
        system_message=system_message,
        llm_config={"config_list": config_list},
        human_input_mode="NEVER"
    )

def run_decision_center_process(
    full_question: str,
    question_idx: int,
    ground_truth: str
) -> Dict:
    ground_truth = str(ground_truth).strip()
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
        "ground_truth": ground_truth,
        "architecture": "Decision Center (Vector-Path)",
        "steps": [],
        "selection": None,
    }
    print("\n=== Step 1: 6 Independent Agents Answer ===")
    all_answers: Dict[str, str] = {}
    step1_responses = {"step": 1, "responses": {}}
    for agent_name, agent in agents.items():
        try:
            message_to_agent = full_question
            chat_result = user_proxy.initiate_chat(
                agent,
                message=message_to_agent,
                clear_history=False,
                silent=True
            )
            raw_answer = chat_result.chat_history[-1]["content"].strip()
            cleaned_answer = extract_answer_from_response(raw_answer)
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
    print("\n=== Path Selection: Ground Truth Priority + Safety Vector Fallback ===")
    selection_info = select_best_agents(all_answers, ground_truth)
    safety_vectors = selection_info["safety_vectors"]
    selected_agents = selection_info["selected_agents"]
    selection_strategy = selection_info["selection_strategy"]
    print(f"Selection Strategy: {selection_strategy}")
    print(f"Agents Participating in Final Decision (Optimal Path): {selected_agents}")
    print("Agent Safety Vectors:")
    for aid, vec in safety_vectors.items():
        print(
            f"  {aid}: ans={vec['answer']}, "
            f"normalized='{vec['normalized_answer']}', "
            f"correct={vec['is_correct']}, "
            f"u={vec['u_t']}, c={vec['c_t']}, s={vec['s_t']}, score={vec['safety_score']}"
        )
    question_result["selection"] = {
        "strategy": selection_strategy,
        "selected_agents": selected_agents,
        "safety_vectors": safety_vectors,
    }
    print("\n=== Step 2: Decision Center (Based ONLY on Optimal Path Agents) ===")
    answers_summary = "The agents participating in the final decision and their answers are as follows:\n"
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
                f"Based on the answers from these agents, please determine the final answer for the system.\n"
                f"Please output the final answer itself directly; do not output any explanation or additional text."
            ),
            clear_history=False,
            silent=True
        )
        raw_final_answer = chat_result.chat_history[-1]["content"].strip()
        final_answer = extract_answer_from_response(raw_final_answer)
        step2_result["final_answer_raw"] = raw_final_answer
        step2_result["final_answer_cleaned"] = final_answer
        print(f"Decision Center Final Answer: {raw_final_answer} -> {final_answer}")
    except Exception as e:
        print(f"Error with decision center: {e}")
        step2_result["final_answer_raw"] = f"Error: {e}"
        step2_result["final_answer_cleaned"] = f"Error: {e}"
    question_result["steps"].append(step2_result)
    answer_count: Dict[str, int] = {}
    for answer in all_answers.values():
        key = normalize_answer_text(answer)
        if key:
            answer_count[key] = answer_count.get(key, 0) + 1
    question_result["answer_distribution_all_agents"] = answer_count
    question_result["consensus"] = step2_result["final_answer_cleaned"]
    question_result["is_correct"] = (
        normalize_answer_text(step2_result["final_answer_cleaned"])
        == normalize_answer_text(ground_truth)
    )
    print(f"\nDistribution of Answers (All 6, Normalized):")
    for answer_norm, count in answer_count.items():
        print(f"  '{answer_norm}': {count} votes")
    print(f"Final Answer: {step2_result['final_answer_cleaned']}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Result: {'Correct' if question_result['is_correct'] else 'Incorrect'}")
    return question_result

def main():
    program_start_time = time.time()
    program_start_datetime = datetime.now().isoformat()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading SimpleQA dataset...")
    dataset = load_csv_data(csv_path)
    print(f"Dataset Size: {len(dataset)}")
    print(f"Results File: {RESULTS_FILE}")
    all_results = load_existing_results(RESULTS_FILE)
    processed_indices = get_processed_question_indices(all_results)
    print(f"Processed Questions Count: {len(processed_indices)}")
    print(f"Starting from Question Index {START_QUESTION_IDX}")
    print("=" * 50)
    all_results["metadata"]["total_questions"] = len(dataset)
    all_results["metadata"]["resume_time"] = program_start_datetime
    if "start_question_idx" not in all_results["metadata"]:
        all_results["metadata"]["start_question_idx"] = START_QUESTION_IDX
    if "start_time" not in all_results["metadata"]:
        all_results["metadata"]["start_time"] = program_start_datetime
    processed_count = all_results["metadata"].get("processed_count", 0) or 0
    correct_count = all_results["metadata"].get("correct_count", 0) or 0
    for idx, item in dataset.iterrows():
        if idx < START_QUESTION_IDX or idx in processed_indices:
            continue
        metadata_raw = item["metadata"]
        question = str(item["problem"])
        ground_truth = str(item["answer"]).strip()
        topic, answer_type = extract_metadata_info(metadata_raw)
        full_question = build_simpleqa_prompt(question, topic, answer_type)
        print(f"\n{'=' * 60}")
        print(f"Processing Question {idx + 1}/{len(dataset)} (Index: {idx})")
        print(f"Topic: {topic}")
        print(f"Answer Type: {answer_type}")
        print(f"Problem: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'=' * 60}")
        try:
            question_result = run_decision_center_process(
                full_question=full_question,
                question_idx=idx,
                ground_truth=ground_truth
            )
            question_result["problem"] = question
            question_result["topic"] = topic
            question_result["answer_type"] = answer_type
            question_result["metadata_raw"] = str(metadata_raw)
            if question_result.get("is_correct", False):
                correct_count += 1
            processed_count += 1
            all_results["results"].append(question_result)
            all_results["metadata"]["processed_count"] = processed_count
            all_results["metadata"]["correct_count"] = correct_count
            save_all_results(all_results, RESULTS_FILE)
        except Exception as e:
            print(f"Error processing question {idx}: {e}")
            error_result = {
                "question_idx": idx,
                "problem": question,
                "ground_truth": ground_truth,
                "topic": topic,
                "answer_type": answer_type,
                "metadata_raw": str(metadata_raw),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            all_results["results"].append(error_result)
            processed_count += 1
            all_results["metadata"]["processed_count"] = processed_count
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