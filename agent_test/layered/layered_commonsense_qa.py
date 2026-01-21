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

config_list = [
    {
        "model": "Your Model",
        "base_url": "Your URL",
        "api_key": "Your Key",
    }
]
topology = {
    "discussant_1": ["discussant_2", "discussant_6"],
    "discussant_2": ["discussant_1", "discussant_3", "discussant_4"],
    "discussant_3": ["discussant_2"],
    "discussant_4": ["discussant_2"],
    "discussant_5": ["discussant_6"],
    "discussant_6": ["discussant_1", "discussant_5"]
}

COMMUNICATION_MODE = "Debate"  #  "Cooperation" | "Debate" | "Competitive"


RESULTS_DIR = "data/commonsense_qa/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "Layered_Debate_results.json")

START_QUESTION_IDX = -1


csv_path = "data/commonsense_qa/commonsense_qa.csv"



def parse_choices(choices_str: str) -> List[str]:
    try:
        choices_str = choices_str.replace("array", "np.array")
        choices_dict = eval(choices_str, {"np": np})
        labels = choices_dict.get("label", [])
        texts = choices_dict.get("text", [])
        if len(labels) == len(texts):
            return [f"{labels[i]}.{texts[i]}" for i in range(len(labels))]
        else:
            print("Warning: The lengths of the labels and texts arrays are inconsistent.")
            return []
    except Exception as e:
        print(f" Failed to parse choices:{e}")
        return []


def load_csv_data(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "rb") as f:
        result = chardet.detect(f.read())
        encoding = result["encoding"]
    df = pd.read_csv(csv_path, encoding=encoding)
    return df

def get_system_message(agent_name: str, neighbors: List[str], mode: str) -> str:
    base_message = f"""You are a participant in the discussion. Your responsibility is to participate in the discussion and help reach the correct answer.

You can only communicate directly with the following neighbors:{', '.join(neighbors)}"""

    mode_specific_messages = {
        "Cooperation": f"""{base_message}
In collaborative mode, you should:

1. Actively share information and perspectives

2. Carefully consider your neighbors' opinions

3. Aim to reach a consensus

4. Maintain a constructive discussion attitude

Please select the correct answer from the given options and output only the letter of the option (A, B, C, D, or E).""",

        "Debate": f"""{base_message}
In debate mode, you should:

1. Firmly defend your viewpoint

2. Provide strong arguments to support your position

3. Critically analyze your neighbor's viewpoint

4. Only change your position when there are sufficient reasons

Please select the correct answer from the given options and output only the letter of the option (A, B, C, D, or E).""",

        "Competitive": f"""{base_message}
In competitive mode, you should:

1. Strive to make your viewpoint the final answer

2. Strategically persuade your neighbor

3. Appropriately emphasize the advantages of your viewpoint

4. The goal is to get the system to adopt your answer. Please select the correct answer from the given options, outputting only the letter of the option (A, B, C, D, or E).""",
    }

    return mode_specific_messages.get(mode, base_message)


def create_agents(mode: str, topology: Dict[str, List[str]]) -> Dict[str, AssistantAgent]:
    agents: Dict[str, AssistantAgent] = {}
    for i in range(1, 7):
        agent_name = f"discussant_{i}"
        neighbors = topology.get(agent_name, [])
        system_message = get_system_message(agent_name, neighbors, mode)
        agent = AssistantAgent(
            name=agent_name,
            system_message=system_message,
            llm_config={"config_list": config_list},
            human_input_mode="NEVER",
        )
        agents[agent_name] = agent
    return agents


def get_discussion_prompt(neighbor_context: str, mode: str, round_num: int) -> str:
    base_prompts = {
        "Cooperation": [
            "Please discuss the correct answer based on your neighbors' opinions. Please only output the letter of the option (A, B, C, D, or E).",
            "Let's continue working together to find the best answer. Please only output the letter of the option (A, B, C, D, or E).",
            "Based on the preceding collaborative discussion, please provide the final answer for our system. Please only output the option letter (A, B, C, D, or E).",
        ],
        "Debate": [
            "Debate with your neighbors and defend your point of view. Please only output the letter of the option (A, B, C, D, or E).",
            "Continue the debate and support your position with reasons. Please only output the letter of your option (A, B, C, D, or E).",
            "After thorough debate, please state your final position. Please only output the letter of your choice (A, B, C, D, or E).",
        ],
        "Competitive": [
            "Please persuade your neighbors to accept your point of view. Please only output the letter of the option (A, B, C, D, or E).",
            "Continue the competition and let your opinion dominate. Please only output the letter of your option (A, B, C, D, or E).",
            "Based on the competitive discussion, please provide the choice you believe should be the system's final answer. Please only output the letter of your choice (A, B, C, D, or E).",
        ],
    }

    mode_prompts = base_prompts.get(mode, base_prompts["Cooperation"])
    prompt = mode_prompts[min(round_num - 1, len(mode_prompts) - 1)]

    if neighbor_context:
        return f"Your neighbors think:\n{neighbor_context}\n\n{prompt}"
    return prompt

def build_question_with_options(question: str, options: List[str]) -> str:
    options_text = "\n".join(options)
    return (
        f"{question}\n\nOptions:\n{options_text}\n\n"
        "Please select the correct answer from the options above and output only the letter of the option (A, B, C, D, or E)."
    )

def load_existing_results(results_file: str) -> Dict:
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f" {e}")

    return {
        "metadata": {
            "communication_mode": COMMUNICATION_MODE,
            "topology": topology,
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
    else:
        return response 


def run_discussion(agents: Dict[str, AssistantAgent], mode: str, full_question: str, question_idx: int, ground_truth: str) -> Dict:
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
        "communication_mode": mode,
        "rounds": []
    }

    round1_responses = {"round": 1, "responses": {}}
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
            round1_responses["responses"][agent_name] = cleaned_answer
            print(f"{agent_name}: {raw_answer} -> {cleaned_answer}")
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
                cleaned_answer = extract_answer_from_response(raw_answer)
                round2_responses["responses"][agent_name] = cleaned_answer
                print(f"{agent_name}: {raw_answer} -> {cleaned_answer}")
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
                cleaned_answer = extract_answer_from_response(raw_answer)
                round3_responses["responses"][agent_name] = cleaned_answer
                print(f"{agent_name}: {raw_answer} -> {cleaned_answer}")
            except Exception as e:
                print(f"Error with {agent_name} in round 3: {e}")
                round3_responses["responses"][agent_name] = f"Error: {e}"
        else:
            round3_responses["responses"][agent_name] = round2_responses["responses"][agent_name]

    question_result["rounds"].append(round3_responses)
    question_result["final_answers"] = round3_responses["responses"].copy()
    final_answer, answer_count = analyze_consensus(round3_responses["responses"], mode)
    question_result["consensus"] = final_answer
    question_result["answer_distribution"] = answer_count

    return question_result


def analyze_consensus(final_answers: Dict[str, str], mode: str) -> Tuple[str, Dict[str, int]]:
    answer_count = {}
    for answer in final_answers.values():
        if answer in ['A', 'B', 'C', 'D', 'E']:
            answer_count[answer] = answer_count.get(answer, 0) + 1

    if answer_count:
        if mode == "Competitive":
            final_answer = max(answer_count.items(), key=lambda x: x[1])[0]
        else:
            final_answer = max(answer_count.items(), key=lambda x: x[1])[0]

        return final_answer, answer_count
    return None, {}


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

        question = item['question']
        choices = item['choices']
        ground_truth = item['answerKey']

        options = parse_choices(choices)
        print(options)
        full_question = build_question_with_options(question, options)



        try:

            agents = create_agents(COMMUNICATION_MODE)


            question_result = run_discussion(agents, COMMUNICATION_MODE, full_question, idx, ground_truth)
            

            if question_result.get("consensus") == ground_truth:
                correct_count += 1
                question_result["is_correct"] = True
            else:
                question_result["is_correct"] = False


            all_results["results"].append(question_result)
            processed_count += 1

            save_all_results(all_results, RESULTS_FILE)



        except Exception as e:
            print(f" {e}")

            error_result = {
                "question_idx": idx,
                "question": question,
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