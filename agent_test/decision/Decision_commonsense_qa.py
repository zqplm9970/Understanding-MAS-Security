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


csv_path = "data/commonsense_qa/commonsense_qa.csv"

RESULTS_DIR = "data/commonsense_qa/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "Decision_Center_results.json")

START_QUESTION_IDX = -1

def parse_choices(choices_str):
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

def build_question_with_options(question: str, options: List[str]) -> str:
    options_text = "\n".join(options)
    return (
        f"{question}\n\nOptions：\n{options_text}\n\n"
        f"Please select the correct answer from the options above and output only the letter of the option (A, B, C, D or E)."
    )
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

    print(f"{results_file}")


def get_processed_question_indices(all_results: Dict) -> set:
    processed_indices = set()
    for result in all_results["results"]:
        if "question_idx" in result:
            processed_indices.add(result["question_idx"])
    return processed_indices


def extract_answer_from_response(response: str) -> str:
    response = response.strip().upper()
    matches = re.findall(r'[A-E]', response)
    
    if matches:
        return matches[0]  
    else:
        return response 


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}"


def create_agents() -> Dict[str, AssistantAgent]:
    agents = {}
    for i in range(1, 7):
        agent_name = f"agent_{i}"

        system_message = f"""You are an independent reasoner. Your responsibilities are:

1. Analyze the problem independently and provide an answer.

2. Do not communicate with other agents.

3. Provide the answer directly without explanation.

4. Answer format: Provide the letter of the option (A, B, C, D, or E).

Please read the question and options carefully and choose the most appropriate answer."""

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

1. Collect answers from a subset of agents (these agents are already selected by the system as being on the "best path").

2. Make decisions solely based on these agents' answers.

3. Analyze the distribution of their answers and use the majority rule to provide the final answer.

4. Provide the final answer directly, without explanation.

For multiple-choice questions, analyze the number of votes for each agent's option and select the option letter (A, B, C, D, or E) that appears most frequently.

If there is a tie, please select the option you deem more reasonable"""

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


    answers_summary = "\n"
    for agent_name, answer in all_answers.items():
        answers_summary += f"{agent_name}: {answer}\n"

    step2_result = {"step": 2, "input_to_decision_center": answers_summary}

    try:
        chat_result = user_proxy.initiate_chat(
            decision_center, 
            message=f"{answers_summary}\nBased on all the answers above, please determine the system's final answer. Please directly output the option letter (A, B, C, D, or E).。",
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
        if answer in ['A', 'B', 'C', 'D', 'E']:
            answer_count[answer] = answer_count.get(answer, 0) + 1

    question_result["answer_distribution"] = answer_count
    question_result["consensus"] = step2_result["final_answer_cleaned"]

    if step2_result["final_answer_cleaned"] == ground_truth:
        question_result["is_correct"] = True
    else:
        question_result["is_correct"] = False

    for answer, count in answer_count.items():
        print(f"  {answer}: {count}票")
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

        question = item['question']
        choices = item['choices']
        ground_truth = item['answerKey']
        
        options = parse_choices(choices)
        full_question = build_question_with_options(question, options)



        try:

            question_result = run_decision_center_process(full_question, idx, ground_truth)
            

            if question_result.get("is_correct", False):
                correct_count += 1
            processed_count += 1


            all_results["results"].append(question_result)

            save_all_results(all_results, RESULTS_FILE)


        except Exception as e:
            print(f"处理问题 {idx} 时出错: {e}")

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