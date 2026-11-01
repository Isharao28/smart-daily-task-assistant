#!/usr/bin/env python3
"""
Comprehensive automated tests for Smart Daily Task Assistant.
Tests TaskManagerTool, TaskAnalysisTool, choose_action (Intent Engine),
and Ollama API integration with Mock HTTP Server.
"""

import os
import sys
import json
import unittest
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import date, timedelta

# Import target functions and classes from main
import main
from main import TaskManagerTool, TaskAnalysisTool, choose_action, is_ollama_available


class MockOllamaHandler(BaseHTTPRequestHandler):
    """Simulates local Ollama API endpoints (/api/tags and /api/generate)."""

    generate_response_payload = {
        "response": json.dumps({
            "action": "add_task",
            "params": {
                "title": "Assignment Submission",
                "priority": "High",
                "deadline": "2026-11-02"
            }
        })
    }

    def do_GET(self):
        if self.path == "/api/tags":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"models": [{"name": "gemma:2b"}]}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            req_body = self.rfile.read(content_length)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.__class__.generate_response_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress HTTP server console logs during unit tests
        pass


class TestSmartDailyTaskAssistant(unittest.TestCase):

    def setUp(self):
        # Create a temporary tasks test file
        self.test_file = os.path.join(os.path.dirname(__file__), "test_tasks.json")
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        self.mgr = TaskManagerTool(self.test_file)
        self.analyzer = TaskAnalysisTool(self.mgr)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_add_and_load_tasks(self):
        """Test adding tasks and verifying JSON persistence."""
        t1 = self.mgr.add_task("Finish OS Lab", priority="High", deadline="2026-11-05")
        self.assertEqual(t1["id"], 1)
        self.assertEqual(t1["title"], "Finish OS Lab")
        self.assertEqual(t1["priority"], "High")
        self.assertEqual(t1["deadline"], "2026-11-05")
        self.assertEqual(t1["status"], "Pending")

        t2 = self.mgr.add_task("Read Chapter 4", priority="Low")
        self.assertEqual(t2["id"], 2)
        self.assertEqual(t2["priority"], "Low")
        self.assertIsNone(t2["deadline"])

        # Reload from file to ensure disk persistence
        fresh_mgr = TaskManagerTool(self.test_file)
        loaded = fresh_mgr.get_all_tasks()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["title"], "Finish OS Lab")

    def test_mark_task_completed(self):
        """Test completing tasks and updating status."""
        t1 = self.mgr.add_task("Submit Assignment", "High")
        success, msg, updated = self.mgr.mark_task_completed(1)
        self.assertTrue(success)
        self.assertEqual(updated["status"], "Completed")

        # Trying to complete again
        success2, msg2, _ = self.mgr.mark_task_completed(1)
        self.assertFalse(success2)

        # Pending vs completed checks
        self.assertEqual(len(self.mgr.get_pending_tasks()), 0)
        self.assertEqual(len(self.mgr.get_completed_tasks()), 1)

    def test_daily_summary(self):
        """Test daily summary generation with urgency detection."""
        today = date.today().strftime("%Y-%m-%d")
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

        self.mgr.add_task("Overdue Task", "High", deadline=yesterday)
        self.mgr.add_task("Due Today Task", "Medium", deadline=today)
        self.mgr.add_task("Completed Task", "Low")
        self.mgr.mark_task_completed(3)

        summary = self.analyzer.get_daily_summary()
        self.assertEqual(summary["total_count"], 3)
        self.assertEqual(summary["completed_count"], 1)
        self.assertEqual(summary["pending_count"], 2)
        self.assertEqual(len(summary["overdue_tasks"]), 1)
        self.assertEqual(len(summary["due_today_tasks"]), 1)
        self.assertEqual(len(summary["high_priority_tasks"]), 1)

    def test_prioritize_tasks_rule_based(self):
        """Test rule-based fallback prioritization ranking."""
        today = date.today().strftime("%Y-%m-%d")
        yesterday = (date.today() - timedelta(days=2)).strftime("%Y-%m-%d")
        future = (date.today() + timedelta(days=10)).strftime("%Y-%m-%d")

        t1 = self.mgr.add_task("Normal Task", "Medium", deadline=future)
        t2 = self.mgr.add_task("Critical Overdue Task", "High", deadline=yesterday)
        t3 = self.mgr.add_task("Due Today", "Low", deadline=today)

        # Force rule-based prioritization
        recommendation = self.analyzer.prioritize_tasks(use_ai=False)
        self.assertIsNotNone(recommendation["recommended_task"])
        # The overdue high priority task should be first
        self.assertEqual(recommendation["recommended_task"]["id"], t2["id"])

    def test_agent_choose_action_intent_recognition(self):
        """Test the agentic intent recognition engine on natural language commands."""
        # 1. Add task
        act1 = choose_action("Add my assignment submission task.")
        self.assertEqual(act1["action"], "add_task")
        self.assertIn("assignment", act1["params"]["title"].lower())

        # 2. View pending tasks
        act2 = choose_action("Show my pending tasks.")
        self.assertEqual(act2["action"], "view_tasks")

        # 3. Prioritize tasks
        act3 = choose_action("Which task should I complete first?")
        self.assertEqual(act3["action"], "prioritize_tasks")

        # 4. Mark task completed
        act4 = choose_action("Mark task 2 as completed.")
        self.assertEqual(act4["action"], "complete_task")
        self.assertEqual(act4["params"].get("task_id"), 2)

        # 5. Daily summary
        act5 = choose_action("Give me my daily summary.")
        self.assertEqual(act5["action"], "daily_summary")

        # 6. Exit
        act6 = choose_action("exit")
        self.assertEqual(act6["action"], "exit")


class TestOllamaIntegration(unittest.TestCase):
    """Tests Ollama API integration using a mock local HTTP server."""

    @classmethod
    def setUpClass(cls):
        # Start mock HTTP server on an ephemeral port
        cls.server = HTTPServer(("127.0.0.1", 0), MockOllamaHandler)
        cls.port = cls.server.server_port
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

        # Temporarily reconfigure main Ollama URL
        cls.original_host = main.OLLAMA_HOST
        cls.original_url = main.OLLAMA_GENERATE_URL
        main.OLLAMA_HOST = f"http://127.0.0.1:{cls.port}"
        main.OLLAMA_GENERATE_URL = f"http://127.0.0.1:{cls.port}/api/generate"
        main._ollama_last_checked = 0.0

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        main.OLLAMA_HOST = cls.original_host
        main.OLLAMA_GENERATE_URL = cls.original_url
        main._ollama_last_checked = 0.0

    def test_ollama_available_and_intent_dispatch(self):
        """Test that Ollama is detected online and intent is resolved via Ollama LLM."""
        self.assertTrue(is_ollama_available())

        # Mock structured response for adding task
        MockOllamaHandler.generate_response_payload = {
            "response": json.dumps({
                "action": "add_task",
                "params": {
                    "title": "Machine Learning Lab Submission",
                    "priority": "High",
                    "deadline": "2026-11-02"
                }
            })
        }

        decision = choose_action("Schedule my ML lab assignment for tomorrow high priority")
        self.assertEqual(decision["action"], "add_task")
        self.assertEqual(decision["params"]["title"], "Machine Learning Lab Submission")
        self.assertIn("Ollama AI", decision["engine"])

    def test_ollama_prioritize_dispatch(self):
        """Test AI prioritization response parsing via Ollama mock."""
        test_file = os.path.join(os.path.dirname(__file__), "mock_tasks.json")
        mgr = TaskManagerTool(test_file)
        t = mgr.add_task("Critical Project", "High", "2026-11-02")
        analyzer = TaskAnalysisTool(mgr)

        MockOllamaHandler.generate_response_payload = {
            "response": json.dumps({
                "task_id": t["id"],
                "task_title": t["title"],
                "reasoning": "High priority and deadline is tomorrow."
            })
        }

        res = analyzer.prioritize_tasks(use_ai=True)
        self.assertIn("Ollama AI", res["source"])
        self.assertEqual(res["recommended_task"]["id"], t["id"])
        self.assertIn("deadline is tomorrow", res["reasoning"])

        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
