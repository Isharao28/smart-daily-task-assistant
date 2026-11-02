#!/usr/bin/env python3
"""
==============================================================================
Project Name   : Smart Daily Task Assistant
Description    : AI-Powered Daily Task Manager using Python, Ollama & Streamlit.
                 Implements Agentic AI principles: Intent Recognition,
                 Action Selection, Tool Execution, and Rule-Based Fallback.
Target Audience: B.Tech CSE students, viva presentation, and portfolio.
==============================================================================
"""

import json
import os
import sys
import re
from datetime import datetime, date, timedelta
import urllib.request
import urllib.error

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_FILE = os.path.join(BASE_DIR, "tasks.json")

# Ollama API configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
# Default to lightweight model, configurable via env var
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma:2b")
OLLAMA_TIMEOUT = 12  # seconds timeout for local AI responses


# ==============================================================================
# OLLAMA CLIENT & UTILITIES
# ==============================================================================

_ollama_last_checked = 0.0
_ollama_is_online = False

def is_ollama_available(timeout: float = 0.5) -> bool:
    """
    Checks whether the local Ollama API server is reachable.
    Caches the result for 3 seconds to avoid redundant socket connections.
    """
    global _ollama_last_checked, _ollama_is_online
    import time
    now = time.time()
    if now - _ollama_last_checked < 3.0:
        return _ollama_is_online

    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ollama_is_online = (resp.status == 200)
    except Exception:
        _ollama_is_online = False

    _ollama_last_checked = now
    return _ollama_is_online


def call_ollama(prompt: str, system_prompt: str = "", timeout: int = OLLAMA_TIMEOUT) -> str | None:
    """
    Sends a generation request to the local Ollama REST API.
    Returns the generated text response, or None if Ollama is unreachable.
    """
    if not is_ollama_available():
        return None

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1  # Low temperature for deterministic, factual outputs
        }
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_GENERATE_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except Exception as e:
        # Fallback will handle cases where Ollama is not running or model is not downloaded
        return None


# ==============================================================================
# TOOL 1: TASK MANAGER TOOL
# ==============================================================================

class TaskManagerTool:
    """
    Tool responsible for Core Task Operations:
    - Adding new tasks
    - Viewing/filtering tasks
    - Marking tasks as completed
    - Persistence in JSON storage (tasks.json)
    """

    def __init__(self, filepath: str = TASKS_FILE):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Creates the JSON file with an empty list if it doesn't exist."""
        if not os.path.exists(self.filepath):
            self.save_tasks([])
        else:
            # Verify file contains valid JSON
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        self.save_tasks([])
                    else:
                        json.loads(content)
            except (json.JSONDecodeError, OSError):
                self.save_tasks([])

    def load_tasks(self) -> list[dict]:
        """Loads all tasks from the JSON file."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return []

    def save_tasks(self, tasks: list[dict]) -> bool:
        """Saves tasks list to JSON file with clean formatting."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=4, ensure_ascii=False)
            return True
        except Exception:
            return False

    def add_task(self, title: str, priority: str = "Medium", deadline: str | None = None) -> dict:
        """
        Adds a new task to tasks.json.
        Parameters:
            title: Name / description of the task
            priority: 'High', 'Medium', or 'Low' (defaults to Medium)
            deadline: Optional date string 'YYYY-MM-DD'
        """
        tasks = self.load_tasks()

        # Generate unique incremental ID
        next_id = max([t["id"] for t in tasks], default=0) + 1

        # Normalize priority
        clean_priority = priority.strip().capitalize() if priority else "Medium"
        if clean_priority not in ["High", "Medium", "Low"]:
            clean_priority = "Medium"

        # Validate & normalize deadline
        clean_deadline = self._normalize_deadline(deadline)

        new_task = {
            "id": next_id,
            "title": title.strip() if title else "New Task",
            "priority": clean_priority,
            "deadline": clean_deadline,
            "status": "Pending",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        tasks.append(new_task)
        self.save_tasks(tasks)
        return new_task

    def get_all_tasks(self) -> list[dict]:
        """Returns all tasks stored in the system."""
        return self.load_tasks()

    def get_pending_tasks(self) -> list[dict]:
        """Returns only tasks that are marked 'Pending'."""
        return [t for t in self.load_tasks() if t.get("status") == "Pending"]

    def get_completed_tasks(self) -> list[dict]:
        """Returns only tasks that are marked 'Completed'."""
        return [t for t in self.load_tasks() if t.get("status") == "Completed"]

    def get_task_by_id(self, task_id: int) -> dict | None:
        """Finds and returns a task by its integer ID."""
        for task in self.load_tasks():
            if task.get("id") == task_id:
                return task
        return None

    def mark_task_completed(self, task_id: int) -> tuple[bool, str, dict | None]:
        """
        Marks the task with the given ID as 'Completed'.
        Returns (success_boolean, message_string, task_dict).
        """
        tasks = self.load_tasks()
        target_task = None

        for task in tasks:
            if task.get("id") == task_id:
                target_task = task
                break

        if not target_task:
            return False, f"Task ID #{task_id} not found.", None

        if target_task.get("status") == "Completed":
            return False, f"Task ID #{task_id} is already marked as Completed.", target_task

        target_task["status"] = "Completed"
        target_task["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_tasks(tasks)
        return True, f"Task ID #{task_id} successfully marked as Completed!", target_task

    def reopen_task(self, task_id: int) -> tuple[bool, str, dict | None]:
        """
        Reopens a completed task, changing its status back to 'Pending'.
        """
        tasks = self.load_tasks()
        target_task = None
        for task in tasks:
            if task.get("id") == task_id:
                target_task = task
                break

        if not target_task:
            return False, f"Task ID #{task_id} not found.", None

        target_task["status"] = "Pending"
        if "completed_at" in target_task:
            del target_task["completed_at"]
        self.save_tasks(tasks)
        return True, f"Task ID #{task_id} reopened to Pending.", target_task

    @staticmethod
    def _normalize_deadline(deadline: str | None) -> str | None:
        """
        Validates and parses optional deadline strings.
        Accepts 'YYYY-MM-DD', 'today', 'tomorrow', or None.
        """
        if not deadline:
            return None
        text = str(deadline).strip().lower()
        today = date.today()
        if text in ["today", "tonight"]:
            return today.strftime("%Y-%m-%d")
        if text in ["tomorrow"]:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")

        # Validate YYYY-MM-DD
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d").date()
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            return None


# ==============================================================================
# TOOL 2: TASK ANALYSIS TOOL
# ==============================================================================

class TaskAnalysisTool:
    """
    Tool responsible for Analytical & Intelligence features:
    - Daily summary generation (total, pending, completed, urgent)
    - Task prioritization using Ollama AI or multi-factor rule-based algorithm
    """

    def __init__(self, task_manager: TaskManagerTool):
        self.task_manager = task_manager

    def get_daily_summary(self) -> dict:
        """
        Compiles high-level metrics about daily tasks:
        - Total, completed, pending counts
        - High-priority pending tasks
        - Urgent pending tasks (overdue, due today, or due tomorrow)
        """
        all_tasks = self.task_manager.get_all_tasks()
        pending = [t for t in all_tasks if t.get("status") == "Pending"]
        completed = [t for t in all_tasks if t.get("status") == "Completed"]

        today = date.today()
        overdue_tasks = []
        due_today_tasks = []
        upcoming_tasks = []
        high_priority_tasks = []

        for t in pending:
            # Check high priority
            if t.get("priority") == "High":
                high_priority_tasks.append(t)

            deadline_str = t.get("deadline")
            if deadline_str:
                try:
                    d_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
                    diff = (d_date - today).days
                    if diff < 0:
                        overdue_tasks.append(t)
                    elif diff == 0:
                        due_today_tasks.append(t)
                    elif diff <= 2:
                        upcoming_tasks.append(t)
                except ValueError:
                    pass

        urgent_tasks = overdue_tasks + due_today_tasks

        return {
            "total_count": len(all_tasks),
            "completed_count": len(completed),
            "pending_count": len(pending),
            "high_priority_tasks": high_priority_tasks,
            "overdue_tasks": overdue_tasks,
            "due_today_tasks": due_today_tasks,
            "upcoming_tasks": upcoming_tasks,
            "urgent_tasks": urgent_tasks,
        }

    def prioritize_tasks(self, use_ai: bool = True) -> dict:
        """
        Analyzes pending tasks and recommends which task to complete first
        with clear reasoning.
        Uses Ollama AI when available, otherwise falls back to a deterministic
        multi-factor scoring algorithm.
        """
        pending = self.task_manager.get_pending_tasks()
        if not pending:
            return {
                "source": "None",
                "recommended_task": None,
                "reasoning": "You have no pending tasks! Great job staying on top of everything."
            }

        # Attempt Ollama AI Prioritization if enabled
        if use_ai:
            ai_result = self._prioritize_with_ollama(pending)
            if ai_result:
                return ai_result

        # Fallback to Rule-Based Prioritization
        return self._prioritize_with_rules(pending)

    def _prioritize_with_ollama(self, pending_tasks: list[dict]) -> dict | None:
        """
        Prompts Ollama to recommend the best next task based on priority and deadline.
        """
        tasks_representation = []
        for t in pending_tasks:
            tasks_representation.append(
                f"- ID: {t['id']}, Title: '{t['title']}', Priority: {t['priority']}, Deadline: {t.get('deadline') or 'None'}"
            )
        task_list_str = "\n".join(tasks_representation)

        system_prompt = (
            "You are an expert AI productivity assistant. Your job is to analyze pending daily tasks "
            "and recommend WHICH SINGLE TASK should be completed first. "
            "Consider that: "
            "1. Overdue tasks or tasks due today have highest urgency. "
            "2. High priority tasks take precedence over Medium and Low. "
            "3. Format your response strictly in JSON: "
            '{"task_id": <int>, "task_title": "<string>", "reasoning": "<concise explanation>"}'
        )

        user_prompt = (
            f"Today's date is: {date.today().strftime('%Y-%m-%d')}\n"
            f"Here are the pending tasks:\n{task_list_str}\n\n"
            f"Which task should the user complete first and why? Respond with JSON only."
        )

        raw_response = call_ollama(prompt=user_prompt, system_prompt=system_prompt, timeout=12)
        if not raw_response:
            return None

        # Try parsing JSON response from Ollama
        try:
            json_text = raw_response
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if match:
                json_text = match.group(0)

            data = json.loads(json_text)
            rec_id = int(data.get("task_id", 0))
            task_obj = self.task_manager.get_task_by_id(rec_id)

            if task_obj and task_obj.get("status") == "Pending":
                return {
                    "source": f"Ollama AI ({OLLAMA_MODEL})",
                    "recommended_task": task_obj,
                    "reasoning": data.get("reasoning", "Recommended based on priority and deadline urgency.")
                }
        except Exception:
            pass

        return None

    def _prioritize_with_rules(self, pending_tasks: list[dict]) -> dict:
        """
        Rule-based fallback algorithm:
        Scores tasks based on Priority weight + Deadline urgency.
        """
        today = date.today()
        scored_tasks = []

        for t in pending_tasks:
            score = 0
            reason_parts = []

            # 1. Priority scoring
            p = t.get("priority", "Medium")
            if p == "High":
                score += 300
                reason_parts.append("it has High Priority")
            elif p == "Medium":
                score += 200
                reason_parts.append("it has Medium Priority")
            else:
                score += 100
                reason_parts.append("it has Low Priority")

            # 2. Deadline scoring
            d_str = t.get("deadline")
            if d_str:
                try:
                    d_date = datetime.strptime(d_str, "%Y-%m-%d").date()
                    days_left = (d_date - today).days

                    if days_left < 0:
                        score += 1000  # Critical overdue
                        reason_parts.insert(0, f"it is OVERDUE by {abs(days_left)} day(s) (due {d_str})")
                    elif days_left == 0:
                        score += 500   # Due today
                        reason_parts.insert(0, f"it is DUE TODAY ({d_str})")
                    elif days_left == 1:
                        score += 350   # Due tomorrow
                        reason_parts.insert(0, f"it is DUE TOMORROW ({d_str})")
                    elif days_left <= 3:
                        score += 250   # Due in 2-3 days
                        reason_parts.insert(0, f"it is due soon in {days_left} days ({d_str})")
                    else:
                        score += 50
                        reason_parts.append(f"deadline is on {d_str}")
                except ValueError:
                    pass
            else:
                reason_parts.append("no deadline specified")

            # 3. ID tie-breaker (earlier created task first)
            score -= t.get("id", 0) * 0.01

            reasoning = "Because " + " and ".join(reason_parts) + "."
            scored_tasks.append((score, t, reasoning))

        # Sort descending by score
        scored_tasks.sort(key=lambda x: x[0], reverse=True)
        best = scored_tasks[0]

        return {
            "source": "Rule-Based Prioritization Engine (Local Fallback)",
            "recommended_task": best[1],
            "reasoning": best[2]
        }


# ==============================================================================
# AGENTIC AI ACTION SELECTION & EXECUTION
# ==============================================================================

# Global tool instances
task_mgr = TaskManagerTool()
task_analyzer = TaskAnalysisTool(task_mgr)


def choose_action(user_instruction: str) -> dict:
    """
    The Agentic Intent Recognition Engine.
    Workflow:
      User Input -> Understand Intent -> Select Action -> Execute Tool

    Uses Ollama structured JSON prompt first.
    If Ollama is unavailable or times out, uses regex/rule-based heuristics.

    Returns a validated action dictionary:
    {
        "action": "add_task | view_tasks | prioritize_tasks | complete_task | daily_summary | exit | unknown",
        "params": { ... },
        "engine": "Ollama LLM" | "Rule-Based Fallback"
    }
    """
    clean_input = user_instruction.strip()
    if not clean_input:
        return {"action": "unknown", "params": {}, "engine": "None"}

    # Attempt AI intent classification via Ollama
    ollama_action = _choose_action_via_ollama(clean_input)
    if ollama_action:
        return ollama_action

    # Fallback to rule-based intent classification
    return _choose_action_via_fallback(clean_input)


def _choose_action_via_ollama(user_input: str) -> dict | None:
    """
    Requests Ollama to map natural language to a defined action schema.
    """
    if not is_ollama_available():
        return None

    system_prompt = (
        "You are an AI Agent Router for a Daily Task Assistant. "
        "Your task is to understand the user's intent and select exactly one action from the following list:\n"
        "1. 'add_task'        -> User wants to add, schedule, or create a task.\n"
        "2. 'view_tasks'      -> User wants to view, show, or list tasks.\n"
        "3. 'prioritize_tasks'-> User asks which task to do first, recommend next task, or prioritize.\n"
        "4. 'complete_task'   -> User wants to mark a task as completed or done.\n"
        "5. 'daily_summary'   -> User wants daily summary, overview, progress, or stats.\n"
        "6. 'exit'            -> User wants to exit, quit, or say goodbye.\n"
        "7. 'unknown'         -> Anything else.\n\n"
        "Extract parameters if applicable: 'title', 'priority' ('High', 'Medium', 'Low'), 'deadline' (YYYY-MM-DD or today/tomorrow), 'task_id' (int).\n"
        "Output ONLY valid JSON matching this format without any other text:\n"
        '{"action": "<action_name>", "params": {"title": null, "priority": null, "deadline": null, "task_id": null}}'
    )

    today_str = date.today().strftime("%Y-%m-%d")
    prompt = f"Today is {today_str}. User input: \"{user_input}\""

    raw_response = call_ollama(prompt=prompt, system_prompt=system_prompt, timeout=10)
    if not raw_response:
        return None

    try:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            action = data.get("action", "").lower().strip()
            allowed = ["add_task", "view_tasks", "prioritize_tasks", "complete_task", "daily_summary", "exit", "unknown"]
            if action in allowed:
                return {
                    "action": action,
                    "params": data.get("params") or {},
                    "engine": f"Ollama AI ({OLLAMA_MODEL})"
                }
    except Exception:
        pass

    return None


def _choose_action_via_fallback(user_input: str) -> dict:
    """
    Deterministic rule-based intent recognition fallback when Ollama is offline.
    """
    text = user_input.strip().lower()

    # 1. Exit intent
    if any(k in text for k in ["exit", "quit", "bye", "goodbye", "close", "stop"]):
        return {"action": "exit", "params": {}, "engine": "Rule-Based Fallback"}

    # 2. Prioritize intent
    if any(k in text for k in ["which task", "what should i do first", "complete first",
                               "do first", "prioritize", "priority recommendation", "what's next"]):
        return {"action": "prioritize_tasks", "params": {}, "engine": "Rule-Based Fallback"}

    # 3. Complete task intent
    if any(k in text for k in ["mark", "complete", "finish", "done"]):
        # Extract task id (e.g., 'task 2', '#2', 'id 2', 'task #2', 'mark 2 as completed')
        id_match = re.search(r"(?:task\s*(?:id)?\s*#?|#)(\d+)|(?:^|\s)(\d+)(?:\s|$)", text)
        task_id = None
        if id_match:
            val = id_match.group(1) or id_match.group(2)
            try:
                task_id = int(val)
            except (ValueError, TypeError):
                task_id = None
        return {
            "action": "complete_task",
            "params": {"task_id": task_id},
            "engine": "Rule-Based Fallback"
        }

    # 4. View tasks intent
    if any(k in text for k in ["show", "view", "list", "display", "all tasks", "pending tasks", "my tasks"]):
        return {"action": "view_tasks", "params": {}, "engine": "Rule-Based Fallback"}

    # 5. Daily summary intent
    if any(k in text for k in ["summary", "daily summary", "overview", "status", "stats", "report", "progress"]):
        return {"action": "daily_summary", "params": {}, "engine": "Rule-Based Fallback"}

    # 6. Add task intent
    if any(k in text for k in ["add", "create", "insert", "new task", "schedule", "remind me"]):
        title = re.sub(r"^(?:please\s+)?(?:add|create|insert|schedule|remind me to)\s+(?:task\s+|a\s+task\s+|my\s+)?", "", text, flags=re.IGNORECASE).strip()
        title = title.rstrip(".")

        # Extract priority if present
        priority = "Medium"
        if "high priority" in text or "urgent" in text:
            priority = "High"
            title = re.sub(r"\b(?:with\s+)?(?:high priority|urgent)\b", "", title, flags=re.IGNORECASE).strip()
        elif "low priority" in text:
            priority = "Low"
            title = re.sub(r"\b(?:with\s+)?low priority\b", "", title, flags=re.IGNORECASE).strip()
        elif "medium priority" in text:
            priority = "Medium"
            title = re.sub(r"\b(?:with\s+)?medium priority\b", "", title, flags=re.IGNORECASE).strip()

        # Extract deadline if present
        deadline = None
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if date_match:
            deadline = date_match.group(1)
            title = re.sub(r"\b(?:due\s+(?:on\s+)?|by\s+)?" + re.escape(deadline) + r"\b", "", title, flags=re.IGNORECASE).strip()
        elif "tomorrow" in text:
            deadline = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
            title = re.sub(r"\b(?:due\s+)?(?:by\s+)?tomorrow\b", "", title, flags=re.IGNORECASE).strip()
        elif "today" in text:
            deadline = date.today().strftime("%Y-%m-%d")
            title = re.sub(r"\b(?:due\s+)?(?:by\s+)?today\b", "", title, flags=re.IGNORECASE).strip()

        # Clean trailing and leading prepositions and noise words
        title = re.sub(r"\s+(?:by|due|with|for)\s*$", "", title, flags=re.IGNORECASE).strip()
        title = re.sub(r"^(?:due|with|by|for)\s+", "", title, flags=re.IGNORECASE).strip()

        if not title:
            title = "New Task"

        return {
            "action": "add_task",
            "params": {"title": title, "priority": priority, "deadline": deadline},
            "engine": "Rule-Based Fallback"
        }

    # Default unknown
    return {"action": "unknown", "params": {}, "engine": "Rule-Based Fallback"}


# ==============================================================================
# ACTION EXECUTION HELPERS
# ==============================================================================

def execute_agent_action(action: str, params: dict) -> dict:
    """
    Executes a chosen action programmatically and returns a structured result.
    Used by Streamlit, CLI, and Web API.
    """
    if action == "add_task":
        title = params.get("title") or "New Task"
        priority = params.get("priority") or "Medium"
        deadline = params.get("deadline")
        task = task_mgr.add_task(title=title, priority=priority, deadline=deadline)
        return {
            "success": True,
            "message": f"Task #{task['id']} ('{task['title']}') added successfully with {task['priority']} priority.",
            "data": task
        }

    elif action == "view_tasks":
        pending = task_mgr.get_pending_tasks()
        completed = task_mgr.get_completed_tasks()
        return {
            "success": True,
            "message": f"Task board: {len(pending)} pending and {len(completed)} completed task(s).",
            "data": {"pending": pending, "completed": completed}
        }

    elif action == "prioritize_tasks":
        res = task_analyzer.prioritize_tasks(use_ai=True)
        if res.get("recommended_task"):
            t = res["recommended_task"]
            msg = f"Recommendation: Complete Task #{t['id']} ('{t['title']}') first. {res.get('reasoning')}"
        else:
            msg = res.get("reasoning", "No pending tasks to prioritize.")
        return {
            "success": True,
            "message": msg,
            "data": res
        }

    elif action == "complete_task":
        task_id = params.get("task_id")
        if task_id is None:
            return {"success": False, "message": "Please specify the Task ID to mark as completed.", "data": None}
        success, msg, task = task_mgr.mark_task_completed(task_id)
        return {"success": success, "message": msg, "data": task}

    elif action == "daily_summary":
        summary = task_analyzer.get_daily_summary()
        msg = f"Summary: {summary['total_count']} total, {summary['completed_count']} completed, {summary['pending_count']} pending, {len(summary['urgent_tasks'])} urgent."
        return {"success": True, "message": msg, "data": summary}

    elif action == "exit":
        return {"success": True, "message": "Goodbye! Have a productive day ahead.", "data": None}

    else:
        return {
            "success": False,
            "message": "I could not understand that intent. Try asking to add, view, complete, or prioritize tasks.",
            "data": None
        }


# ==============================================================================
# STREAMLIT WEB APPLICATION DASHBOARD
# ==============================================================================

def render_streamlit_app():
    """
    Renders the modern, full-featured Streamlit Web Dashboard.
    Replaces blocking terminal input/print with rich Streamlit widgets.
    """
    import streamlit as st

    # 1. Page Configuration
    st.set_page_config(
        page_title="Smart Daily Task Assistant",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Custom CSS styling
    st.markdown("""
        <style>
        .metric-card {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 16px;
            border: 1px solid #334155;
            text-align: center;
        }
        .priority-high {
            color: #ef4444;
            font-weight: 700;
        }
        .priority-med {
            color: #f59e0b;
            font-weight: 700;
        }
        .priority-low {
            color: #3b82f6;
            font-weight: 700;
        }
        .badge-overdue {
            background-color: #7f1d1d;
            color: #fecaca;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-due-today {
            background-color: #78350f;
            color: #fde68a;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize Session State
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []
    if "latest_prioritization" not in st.session_state:
        st.session_state.latest_prioritization = None

    # Load fresh data
    summary = task_analyzer.get_daily_summary()
    pending_tasks = task_mgr.get_pending_tasks()
    completed_tasks = task_mgr.get_completed_tasks()
    ollama_status = is_ollama_available()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ System Status")
        if ollama_status:
            st.success(f"🟢 **Ollama AI Online**\n\nModel: `{OLLAMA_MODEL}`")
        else:
            st.info("🟠 **Rule-Based Fallback Engine**\n\nOffline Mode (Local / Streamlit Cloud)")

        st.caption("Works seamlessly offline with built-in heuristic intent parsing and prioritization.")
        st.divider()

        st.subheader("💡 Smart Agent Examples")
        st.caption("You can paste or click these in **Smart AI Mode**:")
        if st.button("➕ Add DBMS project due tomorrow (High)", use_container_width=True):
            st.session_state.quick_prompt = "Add DBMS project due tomorrow with high priority"
        if st.button("⭐ Which task should I do first?", use_container_width=True):
            st.session_state.quick_prompt = "Which task should I complete first?"
        if st.button("📊 Show daily summary", use_container_width=True):
            st.session_state.quick_prompt = "Give me my daily summary"

        st.divider()
        st.caption(f"📁 Storage: `tasks.json` ({len(task_mgr.get_all_tasks())} total records)")

    # --- MAIN HEADER ---
    st.title("📋 Smart Daily Task Assistant")
    st.write("An Agentic AI assistant that prioritizes activities, tracks daily progress, and executes natural-language commands.")

    # --- TOP METRIC CARDS ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total Tasks", value=summary["total_count"])
    with m2:
        st.metric(label="Pending Tasks", value=summary["pending_count"])
    with m3:
        st.metric(label="Completed Tasks", value=summary["completed_count"])
    with m4:
        st.metric(label="Urgent (Overdue/Today)", value=len(summary["urgent_tasks"]))

    # Urgency Alerts
    if summary["overdue_tasks"]:
        overdue_titles = ", ".join([f"#{t['id']} {t['title']}" for t in summary["overdue_tasks"]])
        st.error(f"🚨 **Overdue Tasks ({len(summary['overdue_tasks'])})**: {overdue_titles}")
    if summary["due_today_tasks"]:
        today_titles = ", ".join([f"#{t['id']} {t['title']}" for t in summary["due_today_tasks"]])
        st.warning(f"⏰ **Due Today ({len(summary['due_today_tasks'])})**: {today_titles}")

    st.divider()

    # --- MAIN NAVIGATION TABS ---
    tab_board, tab_add, tab_prioritize, tab_smart, tab_summary = st.tabs([
        "📋 Task Board",
        "➕ Add Task",
        "⭐ AI Prioritization",
        "🤖 Smart AI Mode",
        "📈 Daily Summary & Analytics"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: TASK BOARD
    # --------------------------------------------------------------------------
    with tab_board:
        st.subheader("Pending Tasks")

        # Priority filter & search
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            p_filter = st.selectbox("Filter by Priority", ["All", "High", "Medium", "Low"], key="p_filter")
        with f_col2:
            search_query = st.text_input("🔍 Search Tasks", placeholder="Search task title...", key="search_query")

        filtered_pending = pending_tasks
        if p_filter != "All":
            filtered_pending = [t for t in filtered_pending if t.get("priority") == p_filter]
        if search_query:
            filtered_pending = [t for t in filtered_pending if search_query.lower() in t.get("title", "").lower()]

        if not filtered_pending:
            if not pending_tasks:
                st.success("🎉 You have no pending tasks! You are all caught up.")
            else:
                st.info("No tasks match your filter criteria.")
        else:
            today_str = date.today().strftime("%Y-%m-%d")
            for t in filtered_pending:
                card = st.container(border=True)
                with card:
                    c1, c2, c3, c4 = st.columns([3, 1.2, 1.5, 1.2])
                    with c1:
                        st.markdown(f"**#{t['id']} - {t['title']}**")
                    with c2:
                        p = t.get("priority", "Medium")
                        color_class = "priority-high" if p == "High" else ("priority-med" if p == "Medium" else "priority-low")
                        st.markdown(f"<span class='{color_class}'>● {p}</span>", unsafe_allow_html=True)
                    with c3:
                        d = t.get("deadline")
                        if not d:
                            st.caption("No deadline")
                        elif d < today_str:
                            st.markdown(f"<span class='badge-overdue'>Overdue: {d}</span>", unsafe_allow_html=True)
                        elif d == today_str:
                            st.markdown(f"<span class='badge-due-today'>Due Today: {d}</span>", unsafe_allow_html=True)
                        else:
                            st.caption(f"Due: {d}")
                    with c4:
                        # Human-in-the-loop confirmation popover
                        with st.popover("Complete"):
                            st.write(f"Mark **Task #{t['id']}** as completed?")
                            if st.button("Yes, Complete", key=f"btn_comp_{t['id']}", type="primary"):
                                task_mgr.mark_task_completed(t['id'])
                                st.toast(f"Task #{t['id']} marked as completed!", icon="✅")
                                st.rerun()

        st.subheader("Completed Tasks")
        if not completed_tasks:
            st.caption("No completed tasks yet.")
        else:
            with st.expander(f"View Completed Tasks ({len(completed_tasks)})", expanded=False):
                for t in completed_tasks:
                    c1, c2, c3, c4 = st.columns([3, 1.2, 1.5, 1.2])
                    with c1:
                        st.markdown(f"~~#{t['id']} - {t['title']}~~")
                    with c2:
                        st.caption(f"Priority: {t.get('priority')}")
                    with c3:
                        st.caption(f"Completed: {t.get('completed_at', 'Done')}")
                    with c4:
                        if st.button("Reopen", key=f"btn_reopen_{t['id']}", help="Mark back to Pending"):
                            task_mgr.reopen_task(t['id'])
                            st.toast(f"Task #{t['id']} reopened!", icon="🔄")
                            st.rerun()

    # --------------------------------------------------------------------------
    # TAB 2: ADD TASK
    # --------------------------------------------------------------------------
    with tab_add:
        st.subheader("Add a New Task")
        with st.form("add_task_form", clear_on_submit=True):
            new_title = st.text_input("Task Title / Description*", placeholder="e.g. Prepare presentation slides for Software Engineering")
            col_p, col_d = st.columns(2)
            with col_p:
                new_priority = st.selectbox("Priority Level", ["High", "Medium", "Low"], index=1)
            with col_d:
                has_deadline = st.checkbox("Set a deadline date?")
                new_deadline = None
                if has_deadline:
                    selected_date = st.date_input("Deadline Date", min_value=date.today())
                    new_deadline = selected_date.strftime("%Y-%m-%d")

            submit_btn = st.form_submit_button("➕ Add Task to List", use_container_width=True)

            if submit_btn:
                if not new_title.strip():
                    st.error("Task title cannot be empty.")
                else:
                    task = task_mgr.add_task(title=new_title, priority=new_priority, deadline=new_deadline)
                    st.success(f"Task #{task['id']} ('{task['title']}') successfully added!")
                    st.rerun()

    # --------------------------------------------------------------------------
    # TAB 3: AI PRIORITIZATION
    # --------------------------------------------------------------------------
    with tab_prioritize:
        st.subheader("AI Task Prioritization Recommender")
        st.write("Analyzes your pending tasks based on deadlines, urgency, and priority levels to recommend which task to tackle first.")

        p_col1, p_col2 = st.columns([1, 3])
        with p_col1:
            if st.button("🔍 Prioritize Now", type="primary", use_container_width=True):
                with st.spinner("Analyzing tasks with AI / rule heuristics..."):
                    st.session_state.latest_prioritization = task_analyzer.prioritize_tasks(use_ai=True)

        if st.session_state.latest_prioritization:
            rec = st.session_state.latest_prioritization
            st.divider()
            st.markdown(f"**Analysis Engine:** `{rec.get('source')}`")

            task = rec.get("recommended_task")
            if not task:
                st.info(rec.get("reasoning", "No pending tasks to prioritize."))
            else:
                card = st.container(border=True)
                with card:
                    st.success("⭐ **RECOMMENDED TASK TO COMPLETE FIRST**")
                    st.markdown(f"### #{task['id']}: {task['title']}")
                    st.markdown(f"**Priority:** {task['priority']} | **Deadline:** {task.get('deadline') or 'None'}")
                    st.divider()
                    st.markdown(f"**Why this task?**\n\n{rec.get('reasoning')}")

                    with st.popover("Mark as Completed"):
                        st.write(f"Mark **Task #{task['id']}** as completed?")
                        if st.button("Confirm Completion", key="btn_rec_comp", type="primary"):
                            task_mgr.mark_task_completed(task['id'])
                            st.session_state.latest_prioritization = None
                            st.toast(f"Task #{task['id']} completed!", icon="🎉")
                            st.rerun()

    # --------------------------------------------------------------------------
    # TAB 4: SMART AI AGENT MODE
    # --------------------------------------------------------------------------
    with tab_smart:
        st.subheader("Smart AI Agent Mode")
        st.caption("Interact using natural English instructions. The AI agent detects intent, extracts parameters, selects the appropriate tool, and updates your task board.")

        # Quick Prompt injector from sidebar
        preset_prompt = st.session_state.pop("quick_prompt", None)

        # Chat interface
        chat_container = st.container(height=350)
        with chat_container:
            if not st.session_state.agent_history:
                st.info("👋 Ask anything! For example:\n- *'Add assignment submission due tomorrow with high priority'*\n- *'Which task should I complete first?'*\n- *'Mark task 1 as completed'*\n- *'Give me my daily summary'*")
            for msg in st.session_state.agent_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "workflow" in msg:
                        with st.expander("🔍 View Agent Decision Workflow"):
                            wf = msg["workflow"]
                            st.markdown(f"- **1. User Input:** \"{wf.get('input')}\"")
                            st.markdown(f"- **2. Understanding:** Processed via `[{wf.get('engine')}]`")
                            st.markdown(f"- **3. Action Selected:** `{wf.get('action')}`")
                            st.markdown(f"- **4. Extracted Params:** `{wf.get('params')}`")

        # Input box
        user_prompt = st.chat_input("Enter natural-language instruction (e.g. 'Add lab exam revision due tomorrow high priority')...")
        if preset_prompt:
            user_prompt = preset_prompt

        if user_prompt:
            # 1. Record User Message
            st.session_state.agent_history.append({"role": "user", "content": user_prompt})

            # 2. Agent Workflow: choose_action()
            decision = choose_action(user_prompt)
            action = decision.get("action")
            params = decision.get("params") or {}
            engine = decision.get("engine", "Fallback")

            # 3. Tool Execution
            res = execute_agent_action(action, params)

            # 4. Record Agent Message
            workflow_info = {
                "input": user_prompt,
                "engine": engine,
                "action": action,
                "params": params
            }
            st.session_state.agent_history.append({
                "role": "assistant",
                "content": res["message"],
                "workflow": workflow_info
            })
            st.rerun()

    # --------------------------------------------------------------------------
    # TAB 5: DAILY SUMMARY & ANALYTICS
    # --------------------------------------------------------------------------
    with tab_summary:
        st.subheader("Daily Summary & Urgent Tasks Breakdown")
        st.write(f"📅 **Date:** {date.today().strftime('%A, %B %d, %Y')}")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("#### High Priority Pending Tasks")
            high_tasks = summary.get("high_priority_tasks", [])
            if not high_tasks:
                st.caption("No high priority tasks pending.")
            else:
                for t in high_tasks:
                    st.markdown(f"- **#{t['id']}**: {t['title']} *(Deadline: {t.get('deadline') or 'None'})*")

        with s_col2:
            st.markdown("#### Urgent Tasks (Overdue & Due Today)")
            urg = summary.get("urgent_tasks", [])
            if not urg:
                st.caption("No urgent tasks. Everything is on schedule!")
            else:
                for t in urg:
                    d_tag = "🔴 Overdue" if t.get("deadline") < date.today().strftime("%Y-%m-%d") else "🟡 Due Today"
                    st.markdown(f"- **#{t['id']}**: {t['title']} — `{d_tag} ({t['deadline']})`")


# ==============================================================================
# MAIN APPLICATION ENTRY POINT
# ==============================================================================

def is_running_in_streamlit() -> bool:
    """
    Detects whether this script is currently being run by Streamlit's script runner.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def main():
    """
    Main entry point:
    - If executed via `streamlit run main.py`, renders the Streamlit app.
    - If executed via bare `python main.py`, automatically launches Streamlit in browser.
    """
    if is_running_in_streamlit():
        render_streamlit_app()
    else:
        # User executed `python main.py` directly: launch Streamlit CLI
        try:
            import streamlit.web.cli as stcli
            print("\n[Smart Daily Task Assistant] Launching Streamlit web application...")
            sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
            sys.exit(stcli.main())
        except ImportError:
            print("[Error] Streamlit is not installed. Run: pip install streamlit")
            sys.exit(1)


# When executed by Streamlit runner (`streamlit run main.py`), top-level runs
if is_running_in_streamlit():
    render_streamlit_app()
elif __name__ == "__main__":
    main()
