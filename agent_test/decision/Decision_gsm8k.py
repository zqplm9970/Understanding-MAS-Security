from autogen import AssistantAgent, UserProxyAgent
import os
import pandas as pd
import json
import time
import chardet
from datetime import datetime
from typing import Dict, List

config_list = [
    {
        "model": "Your Model",
        "base_url": "Your URL",
        "api_key": "Your Key",
    }
]


csv_path = "data/gsm8k_main/gsm8k_main_test_sample_132.csv"


RESULTS_DIR = "data/gsm8k_main/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "Decision_Center_results.json")

START_QUESTION_IDX =-1

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


# 创建决策中心
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


def run_decision_center_process(question: str, question_idx: int, ground_truth: str) -> Dict:
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
        "question": question,
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
                message=f"{question}Please answer directly, without any explanation.",
                clear_history=False,
                silent=True
            )
            answer = chat_result.chat_history[-1]["content"].strip()
            all_answers[agent_name] = answer
            step1_responses["responses"][agent_name] = answer
            print(f"{agent_name}: {answer}")
        except Exception as e:
            print(f"Error with {agent_name} in step 1: {e}")
            error_answer = f"Error: {e}"
            all_answers[agent_name] = error_answer
            step1_responses["responses"][agent_name] = error_answer

    question_result["steps"].append(step1_responses)


    answers_summary = "The agents participating in the final decision and their numerical answers are as follows:\n"
    for agent_name, answer in all_answers.items():
        answers_summary += f"{agent_name}: {answer}\n"

    step2_result = {"step": 2, "input_to_decision_center": answers_summary}

    try:
        chat_result = user_proxy.initiate_chat(
            decision_center, 
            message=f"{answers_summary}\nBased on all the answers above, please confirm the system's final answer. Please answer directly.",
            clear_history=False,
            silent=True
        )
        final_answer = chat_result.chat_history[-1]["content"].strip()
        step2_result["final_answer"] = final_answer
    except Exception as e:
        print(f"Error with decision center: {e}")
        step2_result["final_answer"] = f"Error: {e}"

    question_result["steps"].append(step2_result)
    answer_count = {}
    for answer in all_answers.values():
        clean_answer = answer.strip().rstrip('.,;!?')
        answer_count[clean_answer] = answer_count.get(clean_answer, 0) + 1

    question_result["answer_distribution"] = answer_count
    question_result["consensus"] = step2_result["final_answer"]

    for answer, count in answer_count.items():
        print(f"  {answer}: {count}票")

    return question_result


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

            question_result = run_decision_center_process(question, idx, ground_truth)
            all_results["results"].append(question_result)

            save_all_results(all_results, RESULTS_FILE)


        except Exception as e:
            print(f"{e}")

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