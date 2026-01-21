from autogen import AssistantAgent, UserProxyAgent
import os
import pandas as pd
import json
import time
import chardet
from datetime import datetime
from typing import Dict, List
import re
import numpy as np

config_list = [
    {
        "model": "Your Model",
        "base_url": "Your URL",
        "api_key": "Your Key",
    }
]

csv_path = "data/SimpleQA/SimpleQA_sample_433.csv"

RESULTS_DIR = "data/SimpleQA/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "Decision_Center_results.json")

START_QUESTION_IDX = -1

def parse_simpleqa_metadata(metadata_str):
    try:
        metadata = eval(metadata_str)
        topic = metadata.get('topic', 'Unknown')
        answer_type = metadata.get('answer_type', 'Unknown')
        return topic, answer_type
    except:
        return "Unknown", "Unknown"

def build_simpleqa_question(problem: str, topic: str, answer_type: str) -> str:
    return f"topic: {topic}\ answer type: {answer_type}\nproblem: {problem}\n\nPlease provide the answer directly."

def load_csv_data(csv_path: str):
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read()) 
        encoding = result['encoding'] 
    
    df = pd.read_csv(csv_path, encoding=encoding)
    return df


def load_existing_results(results_file: str) -> Dict:

    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"{e}")

    return {
        "metadata": {
            "architecture": "Decision Center",
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



def get_processed_question_indices(all_results: Dict) -> set:

    processed_indices = set()
    for result in all_results["results"]:
        if "question_idx" in result:
            processed_indices.add(result["question_idx"])
    return processed_indices


def extract_answer_from_response(response: str) -> str:

    response = response.strip()
  
    if len(response) > 100:
        return response[:100] + "..."
    return response


def normalize_answer(answer: str) -> str:

    if not answer:
        return ""

    answer = answer.lower()

    answer = re.sub(r'[^\w\s]', '', answer)

    answer = ' '.join(answer.split())
    return answer


def format_duration(seconds: float) -> str:

    if seconds < 60:
        return f"{seconds:.2f}"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}"


# 创建6个独立智能体
def create_agents() -> Dict[str, AssistantAgent]:
    agents = {}
    for i in range(1, 7):
        agent_name = f"agent_{i}"

        system_message = """You are the Decision Center. Your responsibilities are:
1. Collect answers from a subset of agents (these are the "agents on the optimal path" selected by the system).
2. Make decisions based ONLY on the answers from these agents.
3. Analyze the distribution of their answers and use the majority rule to provide the final answer.
4. Provide the final answer directly; do not explain.

For open-ended Q&A, please analyze the answer text from these agents and select the most frequent or semantically consistent answer.
Output Requirement: Output ONLY the final answer itself; do not output any explanation or additional text."""

        agent = AssistantAgent(
            name=agent_name,
            system_message=system_message,
            llm_config={"config_list": config_list},
            human_input_mode="NEVER"
        )
        agents[agent_name] = agent
    return agents


# 创建决策中心
def create_decision_center() -> AssistantAgent:
    system_message = """You are the decision center. Your responsibilities are:

1. Collect the answers from all independent agents.

2. Analyze the content of the answers.

3. Determine the final answer based on consensus principles.

4. Provide the final answer directly, without explanation.

For open-domain problems, analyze the content of each answer and select the most reasonable and consistent one.

If multiple answers are similar, select the answer that appears most frequently.

If the answers differ significantly, select the most appropriate answer based on your judgment."""

    return AssistantAgent(
        name="decision_center",
        system_message=system_message,
        llm_config={"config_list": config_list},
        human_input_mode="NEVER"
    )


def run_decision_center_process(full_question: str, question_idx: int, ground_truth: str) -> Dict:
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
        "architecture": "Decision Center",
        "steps": []
    }

    all_answers = {}
    step1_responses = {"step": 1, "responses": {}}

    for agent_name, agent in agents.items():
        try:
            chat_result = user_proxy.initiate_chat(
                agent,
                message=f"{full_question}",
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


    answers_summary = "The answers from all agents are as follows:\n"
    for agent_name, answer in all_answers.items():
        answers_summary += f"{agent_name}: {answer}\n"

    step2_result = {"step": 2, "input_to_decision_center": answers_summary}

    try:
        chat_result = user_proxy.initiate_chat(
            decision_center, 
            message=f"{answers_summary}\nBased on all the answers above, please determine the system's final answer. Please directly output the answer you think is the most appropriate.",
            clear_history=False,
            silent=True
        )
        raw_final_answer = chat_result.chat_history[-1]["content"].strip()
        final_answer = extract_answer_from_response(raw_final_answer)
        step2_result["final_answer_raw"] = raw_final_answer
        step2_result["final_answer_cleaned"] = final_answer
    except Exception as e:
        print(f"Error with decision center: {e}")
        step2_result["final_answer_raw"] = f"Error: {e}"
        step2_result["final_answer_cleaned"] = f"Error: {e}"

    question_result["steps"].append(step2_result)

    answer_count = {}
    for answer in all_answers.values():
        if answer not in ['Error:'] and not answer.startswith('Error:'):
            norm_answer = normalize_answer(answer)
            if norm_answer:
                answer_count[norm_answer] = answer_count.get(norm_answer, 0) + 1

    question_result["answer_distribution"] = answer_count
    question_result["consensus"] = step2_result["final_answer_cleaned"]

    normalized_final = normalize_answer(step2_result["final_answer_cleaned"])
    normalized_ground_truth = normalize_answer(ground_truth)
    
    if normalized_final and normalized_ground_truth:
        question_result["is_correct"] = (normalized_final == normalized_ground_truth)
    else:
        question_result["is_correct"] = False


    for answer, count in answer_count.items():
        print(f"  {answer}: {count}")

    return question_result


def main():

    program_start_time = time.time()
    program_start_datetime = datetime.now().isoformat()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = load_csv_data(csv_path)

    all_results = load_existing_results(RESULTS_FILE)
    processed_indices = get_processed_question_indices(all_results)

    if "start_question_idx" not in all_results["metadata"]:
        all_results["metadata"]["start_question_idx"] = START_QUESTION_IDX
    all_results["metadata"]["total_questions"] = len(dataset)
    if "start_time" not in all_results["metadata"]:
        all_results["metadata"]["start_time"] = program_start_datetime

    all_results["metadata"]["resume_time"] = program_start_datetime

    processed_count = 0
    correct_count = 0
    
    for idx, item in dataset.iterrows():
        if idx < START_QUESTION_IDX or idx in processed_indices:
            continue

        problem = item['problem']
        metadata = item['metadata']
        ground_truth = item['answer']
        
        topic, answer_type = parse_simpleqa_metadata(metadata)
        full_question = build_simpleqa_question(problem, topic, answer_type)


        try:
            question_result = run_decision_center_process(full_question, idx, ground_truth)
            
            if question_result.get("is_correct", False):
                correct_count += 1
            processed_count += 1

            all_results["results"].append(question_result)

            save_all_results(all_results, RESULTS_FILE)

        except Exception as e:
            error_result = {
                "question_idx": idx,
                "question": problem,
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