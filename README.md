
---

# Multi-Agent Path Optimization for Hallucination Mitigation

This repository contains the official implementation of the paper **"Multi-Agent Path Optimization for Hallucination Mitigation"**.

## 📄 Abstract

Large Language Models (LLMs) often suffer from hallucinations, which can be propagated or mitigated depending on the interaction architecture in Multi-Agent Systems (MAS). This work introduces a novel **Path Optimization Framework** that dynamically selects optimal communication pathways within various MAS topologies (Centralized, Decentralized, Layered). By optimizing the decision-making trajectory, our method significantly reduces collective hallucinations compared to static baseline systems.

We provide extensive benchmarks across three distinct hallucination domains (**CommonsenseQA**, **GSM8K**, **SimpleQA**), demonstrating the efficacy of our approach in enhancing system reliability and accuracy.

---
## 📂 Repository Structure
| Directory | Description |
| :--- | :--- |
| **`agent_test/`** | **Baseline Benchmarks** (Static Topologies).<br>Evaluation of standard MAS architectures (Centralized, Decentralized, Layered) without optimization to establish baseline hallucination rates. |
| **`agent_path_optimization/`** | **Ours: Path Optimization**.<br>Implementation of the proposed risk-aware decision mechanism that dynamically adjusts agent interaction paths. |
| **`llm_test/`** | **Single-Agent Baselines**.<br>Intrinsic performance tests for the backbone LLM (e.g., Qwen-2.5) to establish lower bounds. |
| **`analysis/`** | **Visualization & Metrics**.<br>Scripts for generating Pareto fronts, propagation heatmaps, and performance tables found in the paper. |
| **`data/`** | **Datasets & Logs**.<br>Contains raw datasets and generated interaction logs (Textual). Organized by dataset and method (`path_optimization` vs `without_optimization`). |
| **`vector/`** | **Vector Artifacts**.<br>High-dimensional vector representations of agent states used for trajectory and convergence analysis. |

---

## 🚀 Getting Started

### 1. Prerequisites

* **Python**: 3.10+
* **OS**: Linux / macOS / Windows
* **Hardware**: CUDA-enabled GPU (recommended for local inference)

### 2. Installation

We recommend using a virtual environment to ensure a clean dependency tree.

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

> **Note**: Please set your API keys (e.g., `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`) in your environment variables before running experiments. **Do not hardcode keys in the scripts.**

---

## 💻 Reproduction Steps

**Crucial**: All scripts must be executed from the **repository root directory** to ensure relative paths resolve correctly.

### Phase 1: Baselines (Static Topologies)

Evaluate standard MAS architectures to gather comparison data.

```bash
# Example: Centralized Topology on CommonsenseQA
python agent_test/centralized/Centralized_commonsense_qa.py

# Example: Layered Topology on GSM8K
python agent_test/layered/Layered_gsm8k.py

```

### Phase 2: Path Optimization (Proposed Method)

Run the risk-aware optimization to observe hallucination mitigation.

```bash
# Example: Centralized Topology with Optimization on CommonsenseQA
python agent_path_optimization/centralized/centralized_commonsense_qa.py

# Example: Decentralized Topology with Optimization on SimpleQA
python agent_path_optimization/decentralized/decentralized_simpleQA.py

```

### Phase 3: Single-Agent Baselines

(Optional) Verify the intrinsic performance of the backbone model.

```bash
python llm_test/commonsense_test.py

```

---

## 📊 Analysis & Visualization

After generating results in `data/`, use the analysis scripts to reproduce the figures in the paper.

```bash
# Generate Pareto Front (Efficiency vs. Accuracy)
python analysis/Pareto.py

# Generate Propagation Heatmaps
python analysis/fig_show.py

```

Results will be saved in:

* `analysis/results/` (Tables & Charts)
* `analysis/propagation/` (Heatmaps)

---

## 🗂 Data Organization

The `data/` directory follows a strict hierarchy to support the analysis pipeline:

* **`data/<dataset_name>/`**: Root for specific task (e.g., `gsm8k_main`).
* **`path_optimization/`**: Results from our proposed method (JSON/CSV).
* **`without_optimization/`**: Results from baseline static topologies.
* **Raw Data**: Original input files (e.g., `*.csv`, `*.arrow`).



**Example Path**:
`data/commonsense_qa/path_optimization/Centralized_Debate_VectorPath_results.json`

---

## ⚖️ License & Citation

* **License**: This code is released under the **MIT License**.
* **Citation**: Please refer to the anonymous submission ID if citing during the review period.

---
