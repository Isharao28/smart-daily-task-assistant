#!/usr/bin/env python3
"""
==============================================================================
Project Name   : Smart Daily Task Assistant
Description    : AI-Powered Daily Task Manager using Python and Ollama.
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
            print(f"[Warning] Failed to load tasks ({e}). Starting with empty list.")
            return []

    def save_tasks(self, tasks: list[dict]) -> bool:
        """Saves tasks list to JSON file with clean formatting."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[Error] Failed to save tasks to {self.filepath}: {e}")
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
        clean_priority = priority.strip().capitalize()
        if clean_priority not in ["High", "Medium", "Low"]:
            clean_priority = "Medium"

        # Validate & normalize deadline
        clean_deadline = self._normalize_deadline(deadline)

        new_task = {
            "id": next_id,
            "title": title.strip(),
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
            # If invalid format, return None or raw if clean
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
            # Extract JSON substring if wrapped in markdown
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
        # Extract JSON
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
        # Extract title
        # Remove trigger prefix like "add task", "add my", "create"
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

        # Extract deadline if present (e.g. YYYY-MM-DD or 'today' or 'tomorrow')
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
# SEPARATE ACTION FUNCTIONS (EXECUTED BY AGENT)
# ==============================================================================

def action_add_task(params: dict, interactive: bool = False) -> None:
    """Executes the Add Task action using TaskManagerTool."""
    title = params.get("title")
    priority = params.get("priority") or "Medium"
    deadline = params.get("deadline")

    # If parameters were not extracted (or interactive mode), prompt user
    if not title or interactive:
        print("\n--- Add New Task ---")
        title = input("Enter task name: ").strip()
        while not title:
            print("Task name cannot be empty.")
            title = input("Enter task name: ").strip()

        p_input = input("Enter priority (High / Medium / Low) [default: Medium]: ").strip()
        priority = p_input if p_input else "Medium"

        d_input = input("Enter deadline (YYYY-MM-DD or 'today'/'tomorrow', leave empty for None): ").strip()
        deadline = d_input if d_input else None

    new_task = task_mgr.add_task(title=title, priority=priority, deadline=deadline)
    print("\n" + "=" * 50)
    print(" [SUCCESS] Task Added Successfully!")
    print(f" ID       : #{new_task['id']}")
    print(f" Title    : {new_task['title']}")
    print(f" Priority : {new_task['priority']}")
    print(f" Deadline : {new_task['deadline'] or 'No deadline'}")
    print(f" Status   : {new_task['status']}")
    print("=" * 50)


def action_view_tasks(params: dict = None) -> None:
    """Executes the View Tasks action, displaying Pending and Completed separately."""
    all_tasks = task_mgr.get_all_tasks()
    pending = [t for t in all_tasks if t.get("status") == "Pending"]
    completed = [t for t in all_tasks if t.get("status") == "Completed"]

    print("\n" + "=" * 70)
    print("                     TASK BOARD                     ")
    print("=" * 70)

    # 1. Pending Tasks Section
    print(f"\n[PENDING TASKS] ({len(pending)})")
    print("-" * 70)
    if not pending:
        print("  (No pending tasks. You are all caught up!)")
    else:
        print(f" {'ID':<4} | {'Priority':<8} | {'Deadline':<12} | {'Task Title'}")
        print("-" * 70)
        for t in pending:
            p_badge = t.get("priority", "Medium")
            d_str = t.get("deadline") or "-"
            print(f" #{t['id']:<3} | {p_badge:<8} | {d_str:<12} | {t['title']}")

    # 2. Completed Tasks Section
    print(f"\n[COMPLETED TASKS] ({len(completed)})")
    print("-" * 70)
    if not completed:
        print("  (No completed tasks yet.)")
    else:
        print(f" {'ID':<4} | {'Priority':<8} | {'Deadline':<12} | {'Task Title'}")
        print("-" * 70)
        for t in completed:
            p_badge = t.get("priority", "Medium")
            d_str = t.get("deadline") or "-"
            print(f" #{t['id']:<3} | {p_badge:<8} | {d_str:<12} | [DONE] {t['title']}")

    print("=" * 70)


def action_prioritize_tasks(params: dict = None) -> None:
    """Executes the Prioritize Tasks action using TaskAnalysisTool."""
    print("\n[AI Prioritizing Tasks] Analyzing urgency, deadlines and priority...")
    result = task_analyzer.prioritize_tasks(use_ai=True)

    print("\n" + "=" * 60)
    print("               TASK PRIORITIZATION RECOMMENDATION               ")
    print("=" * 60)
    print(f" Analysis Engine : {result['source']}")
    print("-" * 60)

    task = result.get("recommended_task")
    if not task:
        print(f" {result.get('reasoning')}")
    else:
        print(" Recommended Task to Complete FIRST:")
        print(f"   Task ID  : #{task['id']}")
        print(f"   Title    : {task['title']}")
        print(f"   Priority : {task['priority']}")
        print(f"   Deadline : {task.get('deadline') or 'None'}")
        print("-" * 60)
        print(f" Explanation:\n   {result.get('reasoning')}")

    print("=" * 60)


def action_complete_task(params: dict = None, interactive: bool = False) -> None:
    """
    Executes the Complete Task action with human-in-the-loop confirmation.
    """
    params = params or {}
    task_id = params.get("task_id")

    # Prompt if task ID is missing
    if task_id is None or interactive:
        pending = task_mgr.get_pending_tasks()
        if not pending:
            print("\nThere are no pending tasks to complete.")
            return

        print("\nPending tasks:")
        for t in pending:
            print(f"  #{t['id']} - {t['title']} (Priority: {t['priority']})")

        val = input("\nEnter Task ID to mark as completed (or 'c' to cancel): ").strip()
        if val.lower() == "c":
            print("Operation cancelled.")
            return
        try:
            task_id = int(val)
        except ValueError:
            print("[Error] Invalid task ID. Must be a number.")
            return

    # Fetch task details for verification
    task = task_mgr.get_task_by_id(task_id)
    if not task:
        print(f"\n[Error] Task ID #{task_id} not found.")
        return

    if task.get("status") == "Completed":
        print(f"\n[Info] Task ID #{task_id} ('{task['title']}') is already completed.")
        return

    # HUMAN-IN-THE-LOOP CONFIRMATION
    print(f"\nTarget Task: #{task['id']} - '{task['title']}' (Priority: {task['priority']}, Deadline: {task.get('deadline') or 'None'})")
    confirm = input(f"Are you sure you want to mark Task #{task_id} as completed? (y/n): ").strip().lower()

    if confirm in ["y", "yes"]:
        success, msg, updated = task_mgr.mark_task_completed(task_id)
        if success:
            print(f"\n[SUCCESS] {msg}")
        else:
            print(f"\n[Error] {msg}")
    else:
        print("\n[Cancelled] Task status unchanged.")


def action_daily_summary(params: dict = None) -> None:
    """Executes the Daily Summary action using TaskAnalysisTool."""
    summary = task_analyzer.get_daily_summary()

    print("\n" + "=" * 55)
    print("                    DAILY SUMMARY                   ")
    print("=" * 55)
    print(f" Today's Date      : {date.today().strftime('%A, %B %d, %Y')}")
    print(f" Total Tasks       : {summary['total_count']}")
    print(f" Completed Tasks   : {summary['completed_count']}")
    print(f" Pending Tasks     : {summary['pending_count']}")
    print("-" * 55)

    # Highlight Urgent Tasks
    urgent = summary.get("urgent_tasks", [])
    print(f"\n[!] Urgent Tasks (Overdue or Due Today): {len(urgent)}")
    if not urgent:
        print("    None. You have no overdue or due-today tasks.")
    else:
        for t in urgent:
            status_tag = "[OVERDUE]" if t.get("deadline") < date.today().strftime("%Y-%m-%d") else "[DUE TODAY]"
            print(f"    * {status_tag} #{t['id']}: {t['title']} (Deadline: {t['deadline']})")

    # Highlight High-Priority Pending Tasks
    high = summary.get("high_priority_tasks", [])
    print(f"\n[*] High-Priority Pending Tasks: {len(high)}")
    if not high:
        print("    None. No pending high-priority tasks.")
    else:
        for t in high:
            d_str = t.get("deadline") or "No deadline"
            print(f"    * #{t['id']}: {t['title']} (Deadline: {d_str})")

    print("=" * 55)


# ==============================================================================
# SMART AI AGENT INTERFACE
# ==============================================================================

def run_smart_mode() -> None:
    """
    Interactive natural-language conversational mode powered by the AI agent.
    Workflow:
      User Input -> Understand Intent (Ollama/Fallback) -> Select Action ->
      Execute Function/Tool -> Display Result
    """
    ollama_status = "ONLINE" if is_ollama_available() else "OFFLINE (Using Rule-Based Fallback)"
    print("\n" + "=" * 65)
    print("                     SMART AI AGENT MODE                    ")
    print(f" Model Engine : {OLLAMA_MODEL} | Status: {ollama_status}")
    print("=" * 65)
    print("You can speak in natural English! Examples:")
    print(" - 'Add my assignment submission task due tomorrow with high priority'")
    print(" - 'Show my pending tasks'")
    print(" - 'Which task should I complete first?'")
    print(" - 'Mark task 2 as completed'")
    print(" - 'Give me my daily summary'")
    print(" - Type 'back' or 'exit' to return to the main menu.")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n[Smart Assistant] What would you like to do? > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nReturning to main menu.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["back", "menu", "return"]:
            print("Returning to main menu.")
            break

        print(f"\n[Agent Workflow]")
        print(f" 1. User Input    : \"{user_input}\"")

        # Step 2: Intent Understanding & Action Selection
        decision = choose_action(user_input)
        action = decision.get("action")
        params = decision.get("params") or {}
        engine = decision.get("engine", "Unknown")

        print(f" 2. Understanding : Intent processed via [{engine}]")
        print(f" 3. Action Chosen : {action}")
        print(f" 4. Tool Execution: Dispatching to Python tool...")

        # Step 3: Tool Execution
        if action == "add_task":
            action_add_task(params, interactive=False)
        elif action == "view_tasks":
            action_view_tasks(params)
        elif action == "prioritize_tasks":
            action_prioritize_tasks(params)
        elif action == "complete_task":
            action_complete_task(params, interactive=False)
        elif action == "daily_summary":
            action_daily_summary(params)
        elif action == "exit":
            print("\nExiting Smart Mode. Goodbye!")
            break
        else:
            print("\n[Agent] I couldn't recognize that command.")
            print("Try asking me to add a task, show tasks, prioritize tasks, complete a task, or give a summary.")


# ==============================================================================
# MAIN APPLICATION CLI MENU
# ==============================================================================

def print_menu() -> None:
    """Displays the main command-line menu."""
    print("\n" + "=" * 45)
    print("         SMART DAILY TASK ASSISTANT          ")
    print("=" * 45)
    print(" 1. Add Task")
    print(" 2. View All Tasks")
    print(" 3. Prioritize Tasks (AI Recommender)")
    print(" 4. Mark Task as Completed")
    print(" 5. Daily Summary")
    print(" 6. Smart AI Mode (Natural Language Agent)")
    print(" 0. Exit")
    print("=" * 45)


def main() -> None:
    """Main application loop."""
    print("\nWelcome to Smart Daily Task Assistant!")
    print(f"Database: {TASKS_FILE}")

    while True:
        print_menu()
        try:
            choice = input("Enter your choice (0-6): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting application. Have a productive day!")
            sys.exit(0)

        if choice == "1":
            action_add_task({}, interactive=True)
        elif choice == "2":
            action_view_tasks()
        elif choice == "3":
            action_prioritize_tasks()
        elif choice == "4":
            action_complete_task({}, interactive=True)
        elif choice == "5":
            action_daily_summary()
        elif choice == "6":
            run_smart_mode()
        elif choice == "0":
            print("\nThank you for using Smart Daily Task Assistant. Goodbye!")
            sys.exit(0)
        else:
            print("\n[Invalid Option] Please enter a number between 0 and 6.")


if __name__ == "__main__":
    main()
