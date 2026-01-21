
from autogen import AssistantAgent, UserProxyAgent
from datasets import load_from_disk
import os
import pandas as pd
from typing import Dict, List, Tuple
import json
import time
import chardet

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


COMMUNICATION_MODE = "Cooperation"  # "Cooperation", "Debate", "Competitive"

RESULTS_DIR = "data/gsm8k_main/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "Layered_Cooperation_results.json")


START_QUESTION_IDX = -1



csv_path = "data/gsm8k_main/gsm8k_main_test_sample_132.csv"


def load_csv_data(csv_path: str):
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read())  
        encoding = result['encoding']  
    
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
Please answer concisely, providing only the final numerical answer; do not include reasoning processes or unnecessary text.""",

        "Debate": f"""{base_message}
In debate mode, you should:

1. Firmly defend your viewpoint

2. Provide strong arguments to support your position

3. Critically analyze your neighbor's viewpoint

4. Only change your position when there are sufficient reasons
Please answer concisely, providing only the final numerical answer; do not include reasoning or extraneous text.""",

        "Competitive": f"""{base_message}
In competitive mode, you should:

1. Strive to make your viewpoint the final answer.

2. Strategically persuade your neighbors.

3. Appropriately emphasize the advantages of your viewpoint.

4. The goal is to get the system to adopt your answer.
Please answer concisely, only outputting the final numerical answer; do not include reasoning processes or extraneous text.。""",
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
            "Please work together to find the correct numerical answer based on your neighbors' perspectives. Please output only a single numerical value, without showing the reasoning process.",
            "Let's continue collaborating to find the best numerical answer. Please output only a single numerical value, without showing the reasoning process.",
            "Based on the previous cooperative discussion, please provide the final numerical answer for our system. Please output only a single numerical value, without showing the reasoning process.",
        ],
        "Debate": [
            "Please debate with your neighbors to defend the numerical answer you believe is correct. Please output only a single numerical value, without showing the reasoning process.",
            "Continue the debate and support your numerical answer with reasons. Please output only a single numerical value, without showing the reasoning process.",
            "After fully debating, please provide your final numerical answer. Please output only a single numerical value, without showing the reasoning process.",
        ],
        "Competitive": [
            "Please persuade your neighbors to accept your numerical answer. Please output only a single numerical value, without showing the reasoning process.",
            "Continue competing to make your numerical answer dominant. Please output only a single numerical value, without showing the reasoning process.",
            "Based on the competitive discussion, please provide the result you believe should be the system's final numerical answer. Please output only a single numerical value, without showing the reasoning process.",
        ],
    }

    mode_prompts = base_prompts.get(mode, base_prompts["Cooperation"])
    prompt = mode_prompts[min(round_num - 1, len(mode_prompts) - 1)]

    if neighbor_context:
        return f"Your neighbors think:：\n{neighbor_context}\n\n{prompt}"
    return f"{prompt}"


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
            "dataset": "GSM8K test",
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

    print(f"{results_file}")


def run_discussion(agents: Dict[str, AssistantAgent], mode: str, question: str, question_idx: int,
                   ground_truth: str) -> Dict:
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0
    )

    question_result = {
        "question_idx": question_idx,
        "question": question,
        "ground_truth": ground_truth,
        "communication_mode": mode,
        "rounds": []
    }

    round1_responses = {"round": 1, "responses": {}}
    for agent_name, agent in agents.items():
        try:
            chat_result = user_proxy.initiate_chat(
                agent,
                message=f"{question}Please answer directly, without any explanation.",
                clear_history=False,
                silent=True
            )
            answer = chat_result.chat_history[-1]["content"].strip()
            round1_responses["responses"][agent_name] = answer
            print(f"{agent_name}: {answer}")
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
                answer = chat_result.chat_history[-1]["content"].strip()
                round2_responses["responses"][agent_name] = answer
                print(f"{agent_name}: {answer}")
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
                answer = chat_result.chat_history[-1]["content"].strip()
                round3_responses["responses"][agent_name] = answer
                print(f"{agent_name}: {answer}")
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
        clean_answer = answer.strip().rstrip('.,;!?')
        answer_count[clean_answer] = answer_count.get(clean_answer, 0) + 1

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

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = load_csv_data(csv_path)

    all_results = load_existing_results(RESULTS_FILE)
    processed_indices = get_processed_question_indices(all_results)


    if "start_question_idx" not in all_results["metadata"]:
        all_results["metadata"]["start_question_idx"] = START_QUESTION_IDX
    all_results["metadata"]["total_questions"] = len(dataset)
    if "start_time" not in all_results["metadata"]:
        all_results["metadata"]["start_time"] = datetime.now().isoformat()

    all_results["metadata"]["resume_time"] = datetime.now().isoformat()

    for idx, item in dataset.iterrows():
        if idx < START_QUESTION_IDX or idx in processed_indices:
            continue

        question = item['question']
        ground_truth = item.get('answer', 'N/A')

        try:
            agents = create_agents(COMMUNICATION_MODE)

            question_result = run_discussion(agents, COMMUNICATION_MODE, question, idx, ground_truth)
            all_results["results"].append(question_result)

            save_all_results(all_results, RESULTS_FILE)


        except Exception as e:

            error_result = {
                "question_idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            all_results["results"].append(error_result)

            save_all_results(all_results, RESULTS_FILE)


    all_results["metadata"]["end_time"] = datetime.now().isoformat()
    all_results["metadata"]["processed_count"] = len([r for r in all_results["results"] if "error" not in r])


    save_all_results(all_results, RESULTS_FILE)

 

if __name__ == "__main__":
    main()