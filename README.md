# Smart Daily Task Assistant (Streamlit + Python + Ollama)

An AI-powered daily task assistant built using **Streamlit**, **Python 3**, and **Ollama**. The application demonstrates practical **Agentic AI principles**: Intent Recognition, Structured Action Selection, Tool Execution, Human-in-the-Loop Confirmation, and Offline Rule-Based Fallback.

Deployed seamlessly on **Streamlit Cloud** and ready to run locally.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Agentic AI Architecture](#agentic-ai-architecture)
3. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
4. [Local Execution](#local-execution)
5. [Project Structure](#project-structure)
6. [Tools & Main Functions Explanation](#tools--main-functions-explanation)
7. [Running Tests](#running-tests)

---

## Key Features

- **A. Task Board**: Live view of pending and completed tasks with priority badges (`High`, `Medium`, `Low`) and deadline status (Overdue, Due Today, Due Soon).
- **B. Add Tasks**: Create tasks with titles, priorities, and deadlines. Automatically saved to `tasks.json`.
- **C. AI Task Prioritization**: Analyzes pending tasks and recommends which task to complete first with detailed reasoning based on deadline urgency and priority weight.
- **D. Complete & Reopen Tasks**: One-click completion with a confirmation popover; completed tasks can also be reopened.
- **E. Daily Summary & Analytics**: Real-time metrics for total, pending, completed, and urgent tasks, with warning banners for overdue items.
- **F. Smart AI Mode (Natural Language Agent)**: Accepts natural-language instructions (e.g. *"Add assignment submission due tomorrow with high priority"*, *"Which task should I complete first?"*, *"Mark task 2 as completed"*).
- **G. Streamlit Cloud Ready**: When deployed on Streamlit Cloud (where local Ollama is offline), the app automatically uses the built-in rule-based fallback without errors or delay.

---

## Agentic AI Architecture

The assistant executes a 4-step agent workflow:

```text
[User Input] ──> [1. choose_action()] ──> [2. Validate & Parse] ──> [3. Tool Execution] ──> [4. Streamlit UI Update]
                      │                                                   │
             ┌────────┴────────┐                              ┌───────────┴───────────┐
             │                 │                              │                       │
      [Ollama LLM]    [Rule Fallback]                  [TaskManagerTool]      [TaskAnalysisTool]
   (Structured JSON)   (Local Engine)                     (tasks.json)       (Priority/Urgency)
```

---

## Streamlit Cloud Deployment

This repository is configured to run out-of-the-box on **Streamlit Community Cloud**:

1. **Main file path**: `main.py`
2. **Requirements**: `requirements.txt` includes `streamlit>=1.30.0` and `requests>=2.28.0`.
3. **Offline AI Fallback**: Streamlit Cloud does not have local Ollama installed. The assistant detects this automatically (< 1ms) and runs the multi-factor rule-based prioritization and regex intent engine.

---

## Local Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with Streamlit
```bash
streamlit run main.py
```
*(Alternatively, running `python main.py` will automatically launch the Streamlit app in your default browser).*

### 3. (Optional) Run with Local Ollama
To enable local LLM inference instead of the offline fallback:
```bash
ollama pull gemma:2b
ollama serve
```
The application will detect Ollama running on `http://localhost:11434` and switch to the LLM engine automatically.

---

## Project Structure

```text
smart-daily-task-assistant/
│
├── main.py              # Streamlit Web Application, Agent Router, and Tools
├── tasks.json           # JSON database for persisting tasks
├── requirements.txt     # Python package requirements (streamlit, requests)
├── test_app.py          # Automated unit & integration test suite
└── README.md            # Project documentation and deployment guide
```

---

## Tools & Main Functions Explanation

### 1. `choose_action(user_instruction)`
- Parses natural language commands into a structured action (`add_task`, `view_tasks`, `prioritize_tasks`, `complete_task`, `daily_summary`, `exit`).
- Uses Ollama structured JSON when online, and falls back to `_choose_action_via_fallback()` when offline.

### 2. `TaskManagerTool`
- `add_task(title, priority, deadline)`: Adds a new task with auto-incremented ID and saves to `tasks.json`.
- `get_pending_tasks()` / `get_completed_tasks()`: Filters tasks by status.
- `mark_task_completed(task_id)`: Marks task as completed with completion timestamp.
- `reopen_task(task_id)`: Reopens a completed task.

### 3. `TaskAnalysisTool`
- `get_daily_summary()`: Aggregates counts and identifies overdue, due today, and high-priority tasks.
- `prioritize_tasks(use_ai=True)`: Recommends which task to complete first using Ollama or the multi-factor scoring formula:
  $$\text{Score} = \text{Urgency Weight (Overdue: 1000, Today: 500, Soon: 250)} + \text{Priority Weight (High: 300, Med: 200, Low: 100)}$$

---

## Running Tests

Run the comprehensive automated test suite (includes mock Ollama API testing):
```bash
python test_app.py
```

Expected output:
```text
test_ollama_available_and_intent_dispatch ... ok
test_ollama_prioritize_dispatch ... ok
test_add_and_load_tasks ... ok
test_agent_choose_action_intent_recognition ... ok
test_daily_summary ... ok
test_mark_task_completed ... ok
test_prioritize_tasks_rule_based ... ok

Ran 7 tests in 1.5s
OK
```
