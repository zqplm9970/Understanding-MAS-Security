from autogen import AssistantAgent, UserProxyAgent
from datasets import load_from_disk  
import os
import pandas as pd
from typing import Dict, List, Tuple
import json
import time
import chardet
import re
import ast
import numpy as np
from datetime import datetime
from collections import Counter

config_list = [
    {
        "model": "Your Model",
        "base_url": "Your URL",
        "api_key": "Your Key",
    }
]

topology = {
    "discussant_1": ["discussant_2", "discussant_5"],
    "discussant_2": ["discussant_1", "discussant_3", "discussant_6"],
    "discussant_3": ["discussant_2", "discussant_4", "discussant_6"],
    "discussant_4": ["discussant_3", "discussant_5", "discussant_6"],
    "discussant_5": ["discussant_1", "discussant_4", "discussant_6"],
    "discussant_6": ["discussant_1","discussant_2", "discussant_3","discussant_4", "discussant_5"]
}


COMMUNICATION_MODE = "Competitive"


csv_path = "data/SimpleQA/SimpleQA_sample_433.csv"
RESULTS_DIR = "data/SimpleQA/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "Centralized_Competitive_results.json")


START_QUESTION_IDX = -1



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
    return (str(topic).strip() if topic else "Other", 
            str(answer_type).strip() if answer_type else "Other")


def normalize_text(s: str) -> str:
  
    if s is None:
        return ""
    s = str(s).strip()

    s = s.strip('"').strip("'")

    s = s.rstrip(" .,:;!?")

    s = re.sub(r"\s+", " ", s)

    return s.lower()


def get_system_message(agent_name: str, neighbors: List[str], mode: str) -> str:
  
    base_message = f"""You are a participant in a discussion. Your role is to engage in the discussion and help reach the correct answer.
You may only communicate directly with the following neighbors: {', '.join(neighbors)}"""

    mode_specific_messages = {
        "Cooperation": f"""
{base_message}
In Cooperation mode, you should:
1. Actively share information and perspectives.
2. Carefully consider your neighbors' opinions.
3. Aim to reach a consensus.
Output Requirement: Output ONLY the final answer. Do not provide explanations or extra characters.""",

        "Debate": f"""
{base_message}
In Debate mode, you should:
1. Firmly defend your own viewpoint.
2. Provide strong arguments to support your stance (in your internal reasoning).
3. Critically analyze your neighbors' viewpoints.
Output Requirement: Output ONLY the final answer. Do not provide explanations or extra characters.""",

        "Competitive": f"""
{base_message}
In Competitive mode, you should:
1. Strive to make your answer the final accepted answer.
2. Strategically persuade your neighbors (in your internal reasoning).
3. Emphasize the advantages of your answer.
Output Requirement: Output ONLY the final answer. Do not provide explanations or extra characters."""
    }
    return mode_specific_messages.get(mode, base_message)


def create_agents(mode: str) -> Dict[str, AssistantAgent]:
    agents = {}
    for i in range(1, 7):
        agent_name = f"discussant_{i}"
        neighbors = topology[agent_name]
        system_message = get_system_message(agent_name, neighbors, mode)
        agent = AssistantAgent(
            name=agent_name,
            system_message=system_message,
            llm_config={"config_list": config_list},
            human_input_mode="NEVER"
        )
        agents[agent_name] = agent
    return agents


def get_discussion_prompt(neighbor_context: str, mode: str, round_num: int) -> str:

    base_prompts = {
        "Cooperation": [
            "Based on the problem and your neighbors' answers, please cooperate to provide the most likely correct answer. Output ONLY the answer itself; do not explain.",
            "Continue collaborating and focus on consistent candidate answers. Output ONLY the answer itself; do not explain.",
            "Synthesizing the previous cooperation, provide the answer you believe should be the system's final output. Output ONLY the answer itself; do not explain.",
        ],
        "Debate": [
            "Based on the problem and type, provide your most confident answer. Output ONLY the answer itself; do not explain.",
            "Continue to maintain or revise your stance. Output ONLY the answer itself; do not explain.",
            "After the debate, provide your final answer. Output ONLY the answer itself; do not explain.",
        ],
        "Competitive": [
            "Please persuade your neighbors and provide your most confident answer. Output ONLY the answer itself; do not explain.",
            "Continue competing and converge on your claim. Output ONLY the answer itself; do not explain.",
            "Based on the competitive discussion, provide the answer you believe should be the system's final output. Output ONLY the answer itself; do not explain.",
        ],
    }
    mode_prompts = base_prompts.get(mode, base_prompts["Cooperation"])
    prompt = mode_prompts[min(round_num - 1, len(mode_prompts) - 1)]

    if neighbor_context:
        return f"Your neighbors' answers:\n{neighbor_context}\n\n{prompt}"
    return prompt


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

    return {
        "metadata": {
            "communication_mode": COMMUNICATION_MODE,
            "topology": topology,
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


def analyze_consensus(final_answers: Dict[str, str], mode: str) -> Tuple[str, Dict[str, int]]:
    norm_map = {}
    for k, v in final_answers.items():
        norm_map[k] = normalize_text(v)

    counter = Counter(norm_map.values())
    if not counter:
        return None, {}


    most_common_norm, _ = counter.most_common(1)[0]

    for agent, norm_val in norm_map.items():
        if norm_val == most_common_norm:
            final_answer_raw = final_answers[agent]
            break


    return final_answer_raw, dict(counter)


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


def run_discussion(agents: Dict[str, AssistantAgent], mode: str, full_prompt: str,
                   question_idx: int, ground_truth: str, topic: str, answer_type: str,
                   problem: str, metadata_raw) -> Dict:

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0
    )

    question_result = {
        "question_idx": question_idx,
        "problem": problem,
        "topic": topic,
        "answer_type": answer_type,
        "prompt": full_prompt,
        "ground_truth": ground_truth,
        "communication_mode": mode,
        "metadata_raw": str(metadata_raw),
        "rounds": []
    }


    round1_responses = {"round": 1, "responses": {}}
    for agent_name, agent in agents.items():
        try:
            chat_result = user_proxy.initiate_chat(
                agent,
                message=full_prompt,
                clear_history=False,
                silent=True
            )
            raw_answer = chat_result.chat_history[-1]["content"].strip()
            round1_responses["responses"][agent_name] = raw_answer
            print(f"{agent_name}: {raw_answer} ")
        except Exception as e:
            print(f"Error with {agent_name} in round 1: {e}")
            round1_responses["responses"][agent_name] = f"Error: {e}"

    question_result["rounds"].append(round1_responses)


    round2_responses = {"round": 2, "responses": {}}
    for agent_name, agent in agents.items():
        neighbors = topology[agent_name]
        neighbor_answers = []
        for neighbor in neighbors:
            if neighbor in round1_responses["responses"]:
                neighbor_answers.append(f"{neighbor}: {round1_responses['responses'][neighbor]}")
        if neighbor_answers:
            neighbor_context = "\n".join(neighbor_answers)
            prompt = get_discussion_prompt(neighbor_context, mode, 2)
            try:
                chat_result = user_proxy.initiate_chat(
                    agent,
                    message=prompt,
                    clear_history=False,
                    silent=True
                )
                raw_answer = chat_result.chat_history[-1]["content"].strip()
                round2_responses["responses"][agent_name] = raw_answer
                print(f"{agent_name}: {raw_answer} ")
            except Exception as e:
                print(f"Error with {agent_name} in round 2: {e}")
                round2_responses["responses"][agent_name] = f"Error: {e}"
        else:
            round2_responses["responses"][agent_name] = round1_responses["responses"][agent_name]
    question_result["rounds"].append(round2_responses)


    round3_responses = {"round": 3, "responses": {}}
    for agent_name, agent in agents.items():
        neighbors = topology[agent_name]
        neighbor_answers = []
        for neighbor in neighbors:
            if neighbor in round2_responses["responses"]:
                neighbor_answers.append(f"{neighbor}: {round2_responses['responses'][neighbor]}")
        if neighbor_answers:
            neighbor_context = "\n".join(neighbor_answers)
            prompt = get_discussion_prompt(neighbor_context, mode, 3)
            try:
                chat_result = user_proxy.initiate_chat(
                    agent,
                    message=prompt,
                    clear_history=False,
                    silent=True
                )
                raw_answer = chat_result.chat_history[-1]["content"].strip()
                round3_responses["responses"][agent_name] = raw_answer
                print(f"{agent_name}: {raw_answer} ")
            except Exception as e:
                print(f"Error with {agent_name} in round 3: {e}")
                round3_responses["responses"][agent_name] = f"Error: {e}"
        else:
            round3_responses["responses"][agent_name] = round2_responses["responses"][agent_name]
    question_result["rounds"].append(round3_responses)

    question_result["final_answers"] = round3_responses["responses"].copy()
    consensus, answer_distribution = analyze_consensus(round3_responses["responses"], mode)
    question_result["consensus"] = consensus
    question_result["answer_distribution"] = answer_distribution
    question_result["is_correct"] = evaluate_correctness(consensus or "", ground_truth or "")

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
            agents = create_agents(COMMUNICATION_MODE)
            question_result = run_discussion(
                agents, COMMUNICATION_MODE, full_prompt,
                idx, ground_truth, topic, answer_type, problem, metadata_raw
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