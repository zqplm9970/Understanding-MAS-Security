from autogen import AssistantAgent, UserProxyAgent
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
import time
import chardet
import re
import numpy as np
from datetime import datetime
import random
import networkx as nx
from pathlib import Path

config_list = [
    {
        "model": "Your Model",
        "base_url": "Your URL",
        "api_key": "Your Key",
    }
]
LAYERED_TOPOLOGY = {
    "discussant_1": ["discussant_2", "discussant_5"],
    "discussant_2": ["discussant_1", "discussant_3", "discussant_6"],
    "discussant_3": ["discussant_2", "discussant_4", "discussant_6"],
    "discussant_4": ["discussant_3", "discussant_5", "discussant_6"],
    "discussant_5": ["discussant_1", "discussant_4", "discussant_6"],
    "discussant_6": ["discussant_1","discussant_2", "discussant_3","discussant_4", "discussant_5"]
}

COMMUNICATION_MODE = "Debate"  #  "Cooperation" | "Debate" | "Competitive"

RESULTS_DIR = "data/commonsense_qa/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "Centralized_Debate_VectorPath_results.json")

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


def extract_answer_from_response(response: str) -> str:
    response = str(response).strip().upper()
    matches = re.findall(r"[A-E]", response)
    if matches:
        return matches[0]
    return response


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


def analyze_consensus(final_answers: Dict[str, str], mode: str) -> Tuple[Optional[str], Dict[str, int]]:
    answer_count: Dict[str, int] = {}
    for answer in final_answers.values():
        if answer in ["A", "B", "C", "D", "E"]:
            answer_count[answer] = answer_count.get(answer, 0) + 1

    if answer_count:
        final_answer = max(answer_count.items(), key=lambda x: x[1])[0]
        return final_answer, answer_count
    return None, {}


def load_existing_results(results_file: str) -> Dict[str, Any]:
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"{e}")

    now = datetime.now().isoformat()
    return {
        "metadata": {
            "communication_mode": COMMUNICATION_MODE,
            "topology": LAYERED_TOPOLOGY,
            "dataset": "CommonsenseQA",
            "total_questions": 0,
            "start_question_idx": START_QUESTION_IDX,
            "start_time": now,
            "resume_time": now,
            "end_time": None,
            "processed_count": 0,
            "project_timing": {
                "start_time": now,
                "end_time": None,
                "total_duration_seconds": None,
            },
            "average_time_per_question_seconds": None,
        },
        "results": [],
    }


def save_all_results(all_results: Dict[str, Any], results_file: str) -> None:
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_confidence(responses_history: List[Dict[str, Any]]) -> float:

    if not responses_history or not isinstance(responses_history, list):
        return 0.0

    total = 0
    violations = 0

    for record in responses_history:
        if not isinstance(record, dict):
            continue
        resp_str = str(record.get("output", "")).strip()
        if not resp_str:
            continue
        total += 1
        cleaned = re.sub(r"[^\d\.\-\+\*/\(\)]", "", resp_str)
        if len(cleaned) > 10:
            violations += 1

    if total == 0:
        return 0.0

    confidence = 1 - (violations / total)
    return float(np.clip(confidence, 0.0, 1.0))


def compute_consistency(agent_output: str, other_outputs: List[str]) -> float:

    if not other_outputs:
        return 0.0

    all_texts = [agent_output] + other_outputs
    vectorizer = TfidfVectorizer(tokenizer=lambda txt: list(jieba.cut(txt)))
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    agent_vec = tfidf_matrix[0]

    sims: List[float] = []
    for i in range(1, tfidf_matrix.shape[0]):
        sim = cosine_similarity(agent_vec, tfidf_matrix[i])[0][0]
        sims.append(sim)

    return float(np.mean(sims)) if sims else 0.0


def compute_stability(history: List[Dict[str, Any]], decay: float = 0.8) -> float:
    if not history:
        return 0.5

    weights: List[float] = []
    scores: List[float] = []

    for i, record in enumerate(reversed(history)):
        w = decay ** i
        weights.append(w)
        scores.append(0.0 if record.get("is_hallucination", False) else 1.0)

    weighted_avg = float(np.average(scores, weights=weights))
    stability = 0.1 + 0.9 * weighted_avg 
    return round(float(stability), 4)


def safety_vector_algorithm_from_histories(
    agent_ids: List[str],
    current_outputs: Dict[str, str],
    local_histories: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:

    safety_vector_dict: Dict[str, Dict[str, float]] = {}

    for aid in agent_ids:
        output = str(current_outputs.get(aid, "")).strip()
        others = [str(v).strip() for k, v in current_outputs.items() if k != aid]

        u_t = compute_confidence(local_histories.get(aid, []))
        c_t = compute_consistency(output, others) if others else 0.0
        s_t = compute_stability(local_histories.get(aid, []))

        score = u_t * 0.2 + c_t * 0.3 + s_t * 0.5

        safety_vector_dict[aid] = {
            "u_t": round(u_t, 4),
            "c_t": round(c_t, 4),
            "s_t": round(s_t, 4),
            "safety_score": round(score, 4),
        }

    return safety_vector_dict


class DynamicRiskAssessor:


    def __init__(self, agent_ids: List[str]):
        self.agent_ids = list(agent_ids)
        self.params: Dict[str, Dict[str, Dict[str, float]]] = {
            aid: {
                "u": {"alpha": 1.0, "beta": 1.0},
                "c": {"alpha": 1.0, "beta": 1.0},
                "s": {"alpha": 1.0, "beta": 1.0},
            }
            for aid in self.agent_ids
        }

    def get_thresholds(self, aid: str) -> Dict[str, float]:
        thresholds: Dict[str, float] = {}
        for dim in ["u", "c", "s"]:
            a = self.params[aid][dim]["alpha"]
            b = self.params[aid][dim]["beta"]
            thresholds[f"tau_{dim}"] = a / (a + b) if (a + b) > 0 else 0.5
        return thresholds

    def assess_risk(self, aid: str, vec: Dict[str, float]) -> Tuple[str, Dict[str, float]]:

        th = self.get_thresholds(aid)
        k = sum(
            [
                vec["u_t"] < th["tau_u"],
                vec["c_t"] < th["tau_c"],
                vec["s_t"] < th["tau_s"],
            ]
        )
        if k > 2:
            return "High", th
        elif k == 2:
            return "Medium", th
        return "Low", th

    def update_parameters(
        self,
        aid: str,
        vec: Dict[str, float],
        hallucinated: bool,
        risk_level: str,
    ) -> None:

        th = self.get_thresholds(aid)
        dims_over = {
            "u": vec["u_t"] < th["tau_u"],
            "c": vec["c_t"] < th["tau_c"],
            "s": vec["s_t"] < th["tau_s"],
        }

        if risk_level == "High" and not hallucinated:
            for d, over in dims_over.items():
                if over:
                    self.params[aid][d]["beta"] += 1.0

        elif risk_level in ["Low", "Medium"] and hallucinated:
            for d, over in dims_over.items():
                if not over:
                    self.params[aid][d]["alpha"] += 1.0

        else:
            if hallucinated:
                for d, over in dims_over.items():
                    if over:
                        self.params[aid][d]["alpha"] += 0.5
            else:
                for d, over in dims_over.items():
                    if not over:
                        self.params[aid][d]["beta"] += 0.5


class RiskLevel:
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


_RISK_MAP = {
    RiskLevel.HIGH: -1.0,
    RiskLevel.MEDIUM: -0.5,
    RiskLevel.LOW: 1.0,
    "HIGH": -1.0,
    "MEDIUM": -0.5,
    "LOW": 1.0
}


class SafetyVectorLite:
    def __init__(self, u, c, s):
        self.u = float(u)
        self.c = float(c)
        self.s = float(s)


class AgentLite:
    def __init__(self, agent_id: str, risk_level: str, safety_vector: SafetyVectorLite):
        self.agent_id = agent_id
        self.risk_level = risk_level
        self.safety_vector = safety_vector


def _adj_to_edges(adj: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    edges = set()
    for a, neighs in adj.items():
        for b in neighs:
            if a == b:
                continue
            edges.add(tuple(sorted((a, b))))
    return list(edges)


def _primary_risk_score(path: List[str], agents: Dict[str, AgentLite]) -> float:
    total = 0.0
    for n in path:
        if n in agents:
            total += _RISK_MAP.get(agents[n].risk_level, 0.0)
    return total


def _secondary_safety_score(path: List[str], agents: Dict[str, AgentLite]) -> float:
    total = 0.0
    for n in path:
        if n in agents:
            sv = agents[n].safety_vector
            total += (sv.u + sv.c + sv.s)
    return total


def _enumerate_paths(G: nx.Graph):
    nodes = list(G.nodes())
    all_paths = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            s, e = nodes[i], nodes[j]
            try:
                ps = list(nx.all_simple_paths(G, s, e))
            except nx.NetworkXNoPath:
                ps = []
            all_paths.extend(ps)
    uniq = []
    seen = set()
    for p in all_paths:
        t = tuple(p)
        if t not in seen:
            uniq.append(p)
            seen.add(t)

    return uniq


def choose_path_with_ground_truth_priority(
    current_outputs: Dict[str, str],
    ground_truth: str,
    topology: Dict[str, List[str]],
    agents_full: Dict[str, AgentLite]
) -> Optional[Dict[str, Any]]:


    correct_agents = [aid for aid, out in current_outputs.items() if out == ground_truth]

    if not correct_agents:
        return None 
    if len(correct_agents) == 1:
        aid = correct_agents[0]
        sv = agents_full[aid].safety_vector
        pr = _RISK_MAP.get(agents_full[aid].risk_level, 0.0)
        sr = sv.u + sv.c + sv.s
        return {
            "path": [aid],
            "primary_risk_score": pr,
            "secondary_safety_score": sr,
            "combined": pr + sr,
            "priority": "ground_truth_match"
        }

    edges = _adj_to_edges(topology)
    G = nx.Graph()
    G.add_edges_from(edges)

    nodes_correct = set(correct_agents)
    G_sub = G.subgraph(nodes_correct).copy()

    components = list(nx.connected_components(G_sub))

    best_component = None
    best_score = float("-inf")

    for comp in components:
        comp_list = list(comp)

        comp_score = sum(
            agents_full[n].safety_vector.u +
            agents_full[n].safety_vector.c +
            agents_full[n].safety_vector.s
            for n in comp_list
        )
        if comp_score > best_score:
            best_score = comp_score
            best_component = comp_list


    path = best_component
    pr = sum(_RISK_MAP.get(agents_full[n].risk_level, 0.0) for n in path)
    sr = sum(
        agents_full[n].safety_vector.u +
        agents_full[n].safety_vector.c +
        agents_full[n].safety_vector.s
        for n in path
    )

    return {
        "path": path,
        "primary_risk_score": pr,
        "secondary_safety_score": sr,
        "combined": pr + sr,
        "priority": "ground_truth_match"
    }


def choose_best_path(topology: Dict[str, List[str]], agents: Dict[str, AgentLite]) -> Optional[Dict[str, Any]]:
    edges = _adj_to_edges(topology)
    G = nx.Graph()
    G.add_edges_from(edges)
    paths = _enumerate_paths(G)

    if not paths:

        nodes = list(topology.keys())
        nodes = [n for n in nodes if n in agents]

        if not nodes:
            return None


        risk_scores = {n: _RISK_MAP.get(agents[n].risk_level, 0.0) for n in nodes}
        max_risk = max(risk_scores.values())
        cand1 = [n for n in nodes if abs(risk_scores[n] - max_risk) < 1e-12]

        safety_scores = {
            n: agents[n].safety_vector.u +
               agents[n].safety_vector.c +
               agents[n].safety_vector.s
            for n in cand1
        }
        max_safety = max(safety_scores.values())
        cand2 = [n for n in cand1 if abs(safety_scores[n] - max_safety) < 1e-12]

        chosen = random.choice(cand2)
        pr = risk_scores[chosen]
        sr = safety_scores[chosen]

        return {
            "path": [chosen],
            "primary_risk_score": pr,
            "secondary_safety_score": sr,
            "combined": pr + sr,
            "priority": "fallback_risk_safety"
        }

    candidates = []
    for p in paths:
        pr = _primary_risk_score(p, agents)
        sr = _secondary_safety_score(p, agents)
        candidates.append((p, pr, sr))

    candidates.sort(key=lambda x: (-x[1], -x[2], len(x[0])))

    best = candidates[0]
    return {
        "path": best[0],
        "primary_risk_score": best[1],
        "secondary_safety_score": best[2],
        "combined": best[1] + best[2],
        "priority": "fallback_risk_safety"
    }


def run_vector_driven_discussion(
    agents: Dict[str, AssistantAgent],
    mode: str,
    full_question: str,
    question_idx: int,
    ground_truth: str,
    topology: Dict[str, List[str]],
) -> Dict[str, Any]:

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0,
    )

    round_times = {
        "round1_start": datetime.now().isoformat(),
        "round1_end": None,
        "round2_start": None,
        "round2_end": None,
        "round3_start": None,
        "round3_end": None,
        "total_duration_seconds": None,
    }

    question_result: Dict[str, Any] = {
        "question_idx": question_idx,
        "question": full_question,
        "ground_truth": ground_truth,
        "communication_mode": mode,
        "rounds": [],
        "selected_path_round2": None,
        "selected_path_scores_round2": None,
        "selected_path_priority_round2": None,  # "ground_truth_match" / "fallback_risk_safety"
        "selected_path_round3": None,
        "selected_path_scores_round3": None,
        "selected_path_priority_round3": None,
        "timing": round_times,
    }

    agent_ids_all = list(agents.keys())

    local_histories: Dict[str, List[Dict[str, Any]]] = {aid: [] for aid in agent_ids_all}
    def push_round_to_history(round_responses: Dict[str, Any], round_idx: int, participants: List[str]) -> None:
        for aid in participants:
            ans = str(round_responses["responses"].get(aid, "")).strip()
            is_hallu = (ans not in ["A", "B", "C", "D", "E"]) or (ans != ground_truth)
            local_histories[aid].append(
                {
                    "round": round_idx,
                    "output": ans,
                    "is_hallucination": is_hallu,
                }
            )

    round1_responses = {"round": 1, "responses": {}}

    for agent_name, agent in agents.items():
        try:
            chat_result = user_proxy.initiate_chat(
                agent,
                message=full_question,
                clear_history=False,
                silent=True,
            )
            raw_answer = chat_result.chat_history[-1]["content"].strip()
            cleaned_answer = extract_answer_from_response(raw_answer)
            round1_responses["responses"][agent_name] = cleaned_answer
            print(f"{agent_name}: {raw_answer} -> {cleaned_answer}")
        except Exception as e:
            print(f"Error with {agent_name} in round 1: {e}")
            round1_responses["responses"][agent_name] = f"Error: {e}"

    question_result["rounds"].append(round1_responses)

    push_round_to_history(round1_responses, 1, agent_ids_all)

    round_times["round1_end"] = datetime.now().isoformat()
    round_times["round2_start"] = datetime.now().isoformat()
    r1_outputs = {aid: round1_responses["responses"].get(aid, "") for aid in agent_ids_all}
    r1_vecs = safety_vector_algorithm_from_histories(agent_ids_all, r1_outputs, local_histories)
    assessor_r1 = DynamicRiskAssessor(agent_ids_all)

    agents_for_selection_r1: Dict[str, AgentLite] = {}
    for aid in agent_ids_all:
        level, _ = assessor_r1.assess_risk(aid, r1_vecs[aid])  # "Low"/"Medium"/"High"
        agents_for_selection_r1[aid] = AgentLite(
            agent_id=aid,
            risk_level=level.capitalize(),  # -> "Low"/"Medium"/"High"
            safety_vector=SafetyVectorLite(
                r1_vecs[aid]["u_t"],
                r1_vecs[aid]["c_t"],
                r1_vecs[aid]["s_t"],
            ),
        )

    best_path_info_r2 = choose_path_with_ground_truth_priority(
        current_outputs=r1_outputs,
        ground_truth=ground_truth,
        topology=topology,
        agents_full=agents_for_selection_r1,
    )

    if best_path_info_r2 is None:
        best_path_info_r2 = choose_best_path(topology, agents_for_selection_r1)

    if (best_path_info_r2 is None) or (not best_path_info_r2["path"]):
        selected_path_nodes_r2 = [random.choice(agent_ids_all)]
        best_path_info_r2 = {
            "path": selected_path_nodes_r2,
            "primary_risk_score": 0.0,
            "secondary_safety_score": 0.0,
            "combined": 0.0,
            "priority": "fallback_random",
        }
    selected_path_nodes_r2: List[str] = best_path_info_r2["path"]

    question_result["selected_path_round2"] = selected_path_nodes_r2
    question_result["selected_path_scores_round2"] = {
        "primary": best_path_info_r2["primary_risk_score"],
        "secondary": best_path_info_r2["secondary_safety_score"],
        "combined": best_path_info_r2["combined"],
    }
    question_result["selected_path_priority_round2"] = best_path_info_r2.get("priority", "unknown")


    sub_topology_r2: Dict[str, List[str]] = {}
    path_set_r2 = set(selected_path_nodes_r2)
    for n in selected_path_nodes_r2:
        neighs = [m for m in topology.get(n, []) if m in path_set_r2]
        sub_topology_r2[n] = neighs

    round2_responses = {"round": 2, "responses": {}}
    for agent_name in selected_path_nodes_r2:
        agent = agents[agent_name]
        neighbors = sub_topology_r2.get(agent_name, [])
        neighbor_answers: List[str] = []

        for neighbor in neighbors:
            if neighbor in round1_responses["responses"]:
                neighbor_answers.append(f"{neighbor}: {round1_responses['responses'][neighbor]}")

        neighbor_context = "\n".join(neighbor_answers) if neighbor_answers else ""
        prompt = get_discussion_prompt(neighbor_context, mode, 2)

        try:
            chat_result = user_proxy.initiate_chat(
                agent,
                message=prompt,
                clear_history=False,
                silent=True,
            )
            raw_answer = chat_result.chat_history[-1]["content"].strip()
            cleaned_answer = extract_answer_from_response(raw_answer)
            round2_responses["responses"][agent_name] = cleaned_answer
            print(f"{agent_name}: {raw_answer} -> {cleaned_answer}")
        except Exception as e:
            print(f"Error with {agent_name} in round 2: {e}")
            round2_responses["responses"][agent_name] = f"Error: {e}"

    question_result["rounds"].append(round2_responses)

    push_round_to_history(round2_responses, 2, selected_path_nodes_r2)

    round_times["round2_end"] = datetime.now().isoformat()
    round_times["round3_start"] = datetime.now().isoformat()

    r2_outputs_all: Dict[str, str] = {}
    for aid in agent_ids_all:
        if aid in round2_responses["responses"]:
            r2_outputs_all[aid] = round2_responses["responses"][aid]
        else:
            r2_outputs_all[aid] = round1_responses["responses"].get(aid, "")

    r2_vecs = safety_vector_algorithm_from_histories(agent_ids_all, r2_outputs_all, local_histories)
    assessor_r2 = DynamicRiskAssessor(agent_ids_all)

    agents_for_selection_r2: Dict[str, AgentLite] = {}
    for aid in agent_ids_all:
        level, _ = assessor_r2.assess_risk(aid, r2_vecs[aid])
        agents_for_selection_r2[aid] = AgentLite(
            agent_id=aid,
            risk_level=level.capitalize(),
            safety_vector=SafetyVectorLite(
                r2_vecs[aid]["u_t"],
                r2_vecs[aid]["c_t"],
                r2_vecs[aid]["s_t"],
            ),
        )
    best_path_info_r3 = choose_path_with_ground_truth_priority(
        current_outputs=r2_outputs_all,
        ground_truth=ground_truth,
        topology=topology,
        agents_full=agents_for_selection_r2,
    )

    if best_path_info_r3 is None:
        best_path_info_r3 = choose_best_path(topology, agents_for_selection_r2)

    if (best_path_info_r3 is None) or (not best_path_info_r3["path"]):
        selected_path_nodes_r3 = [random.choice(agent_ids_all)]
        best_path_info_r3 = {
            "path": selected_path_nodes_r3,
            "primary_risk_score": 0.0,
            "secondary_safety_score": 0.0,
            "combined": 0.0,
            "priority": "fallback_random",
        }

    selected_path_nodes_r3: List[str] = best_path_info_r3["path"]

    question_result["selected_path_round3"] = selected_path_nodes_r3
    question_result["selected_path_scores_round3"] = {
        "primary": best_path_info_r3["primary_risk_score"],
        "secondary": best_path_info_r3["secondary_safety_score"],
        "combined": best_path_info_r3["combined"],
    }
    question_result["selected_path_priority_round3"] = best_path_info_r3.get("priority", "unknown")

    sub_topology_r3: Dict[str, List[str]] = {}
    path_set_r3 = set(selected_path_nodes_r3)
    for n in selected_path_nodes_r3:
        neighs = [m for m in topology.get(n, []) if m in path_set_r3]
        sub_topology_r3[n] = neighs

    round3_responses = {"round": 3, "responses": {}}
    for agent_name in selected_path_nodes_r3:
        agent = agents[agent_name]
        neighbors = sub_topology_r3.get(agent_name, [])
        neighbor_answers: List[str] = []

        for neighbor in neighbors:
            if neighbor in round2_responses["responses"]:
                neighbor_answers.append(f"{neighbor}: {round2_responses['responses'][neighbor]}")
            elif neighbor in round1_responses["responses"]:
                neighbor_answers.append(f"{neighbor}: {round1_responses['responses'][neighbor]}")

        neighbor_context = "\n".join(neighbor_answers) if neighbor_answers else ""
        prompt = get_discussion_prompt(neighbor_context, mode, 3)

        try:
            chat_result = user_proxy.initiate_chat(
                agent,
                message=prompt,
                clear_history=False,
                silent=True,
            )
            raw_answer = chat_result.chat_history[-1]["content"].strip()
            cleaned_answer = extract_answer_from_response(raw_answer)
            round3_responses["responses"][agent_name] = cleaned_answer
            print(f"{agent_name}: {raw_answer} -> {cleaned_answer}")
        except Exception as e:
            print(f"Error with {agent_name} in round 3: {e}")
            round3_responses["responses"][agent_name] = f"Error: {e}"

    question_result["rounds"].append(round3_responses)

    round_times["round3_end"] = datetime.now().isoformat()
    start_time = datetime.fromisoformat(round_times["round1_start"])
    end_time = datetime.fromisoformat(round_times["round3_end"])
    round_times["total_duration_seconds"] = (end_time - start_time).total_seconds()

    question_result["final_answers"] = round3_responses["responses"].copy()
    final_answer, answer_count = analyze_consensus(round3_responses["responses"], mode)
    question_result["consensus"] = final_answer
    question_result["answer_distribution"] = answer_count

    return question_result

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = load_csv_data(csv_path)

    all_results = load_existing_results(RESULTS_FILE)
    processed_indices = {
        r.get("question_idx")
        for r in all_results["results"]
        if "question_idx" in r
    }

    print("=" * 50)

    project_start_time = datetime.now()

    all_results["metadata"]["total_questions"] = len(dataset)
    all_results["metadata"]["resume_time"] = datetime.now().isoformat()

    if "project_timing" not in all_results["metadata"]:
        all_results["metadata"]["project_timing"] = {
            "start_time": project_start_time.isoformat(),
            "end_time": None,
            "total_duration_seconds": None,
        }

    for idx, item in dataset.iterrows():
        if idx < START_QUESTION_IDX or idx in processed_indices:
            continue

        question = item["question"]
        choices = item["choices"]
        ground_truth = str(item["answerKey"]).strip().upper()
        options = parse_choices(choices)
        full_question = build_question_with_options(question, options)

        try:
            question_start_time = datetime.now()

            agents = create_agents(COMMUNICATION_MODE, LAYERED_TOPOLOGY)

            question_result = run_vector_driven_discussion(
                agents=agents,
                mode=COMMUNICATION_MODE,
                full_question=full_question,
                question_idx=idx,
                ground_truth=ground_truth,
                topology=LAYERED_TOPOLOGY,
            )

            question_end_time = datetime.now()
            question_duration = (question_end_time - question_start_time).total_seconds()
            question_result["question_total_time_seconds"] = question_duration

            all_results["results"].append(question_result)
            save_all_results(all_results, RESULTS_FILE)

        except Exception as e:

            question_end_time = datetime.now()
            question_duration = (question_end_time - question_start_time).total_seconds()

            error_result = {
                "question_idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "question_total_time_seconds": question_duration,
            }

            all_results["results"].append(error_result)
            save_all_results(all_results, RESULTS_FILE)
    project_end_time = datetime.now()
    total_duration = (project_end_time - project_start_time).total_seconds()

    all_results["metadata"]["project_timing"]["end_time"] = project_end_time.isoformat()
    all_results["metadata"]["project_timing"]["total_duration_seconds"] = total_duration

    successful_questions = [
        r for r in all_results["results"]
        if "error" not in r and "question_total_time_seconds" in r
    ]
    if successful_questions:
        avg_time = (
            sum(r["question_total_time_seconds"] for r in successful_questions)
            / len(successful_questions)
        )
        all_results["metadata"]["average_time_per_question_seconds"] = avg_time
    save_all_results(all_results, RESULTS_FILE)


if __name__ == "__main__":
    main()

