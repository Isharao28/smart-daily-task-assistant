#!/usr/bin/env python3
"""
==============================================================================
Smart Daily Task Assistant - Web Application Server
Serves a clean, modern web interface on http://localhost:5000
Powered by Python's built-in standard library http.server (no external dependencies).
Integrates directly with main.py (TaskManagerTool, TaskAnalysisTool, choose_action).
==============================================================================
"""

import json
import os
import sys
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import date

# Import tools and AI agent from main.py
import main
from main import task_mgr, task_analyzer, choose_action, is_ollama_available, OLLAMA_MODEL

PORT = 5000

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Daily Task Assistant</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        body { background-color: var(--bg-primary); color: var(--text-primary); line-height: 1.5; padding: 24px; max-width: 1200px; margin: 0 auto; }
        
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
        h1 { font-size: 1.6rem; font-weight: 700; display: flex; align-items: center; gap: 10px; }
        .badge { padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
        .badge-online { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
        .badge-fallback { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }

        /* Summary Cards */
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .summary-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
        .summary-card .label { font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
        .summary-card .val { font-size: 2rem; font-weight: 700; margin-top: 6px; }

        .layout-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        @media (max-width: 900px) { .layout-grid { grid-template-columns: 1fr; } }

        .panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }
        .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
        .panel-title { font-size: 1.15rem; font-weight: 600; }

        /* Forms */
        .form-row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; }
        label { font-size: 0.85rem; color: var(--text-secondary); }
        input, select, textarea {
            background: var(--bg-input); border: 1px solid var(--border); color: #fff;
            padding: 10px 14px; border-radius: 8px; font-size: 0.95rem; outline: none; width: 100%;
        }
        input:focus, select:focus, textarea:focus { border-color: var(--accent-blue); }
        .form-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

        button {
            cursor: pointer; background: var(--accent-blue); color: #fff; border: none;
            padding: 10px 16px; border-radius: 8px; font-weight: 600; font-size: 0.9rem;
            transition: all 0.2s; display: inline-flex; align-items: center; justify-content: center; gap: 6px;
        }
        button:hover { opacity: 0.9; }
        button.btn-success { background: var(--accent-green); }
        button.btn-amber { background: var(--accent-amber); color: #1e293b; }

        /* Task List */
        .task-list { display: flex; flex-direction: column; gap: 10px; }
        .task-item {
            background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border);
            border-radius: 8px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center;
        }
        .task-item.completed { opacity: 0.6; text-decoration: line-through; }
        .task-title { font-weight: 600; font-size: 0.95rem; }
        .task-meta { font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px; display: flex; gap: 12px; }
        .p-High { color: #f87171; font-weight: 600; }
        .p-Medium { color: #fbbf24; font-weight: 600; }
        .p-Low { color: #60a5fa; font-weight: 600; }

        /* AI Recommendation Box */
        .ai-box {
            background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px; padding: 16px; margin-bottom: 20px;
        }
        .ai-box h3 { font-size: 1rem; color: #93c5fd; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }

        /* Chat / Agent Window */
        .chat-history {
            background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border); border-radius: 8px;
            padding: 12px; height: 260px; overflow-y: auto; margin-bottom: 12px; display: flex; flex-direction: column; gap: 10px;
        }
        .msg { padding: 8px 12px; border-radius: 8px; font-size: 0.88rem; max-width: 85%; }
        .msg-user { background: var(--accent-blue); color: #fff; align-self: flex-end; }
        .msg-agent { background: var(--bg-input); color: #e2e8f0; align-self: flex-start; border: 1px solid var(--border); }
        .msg-step { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }
    </style>
</head>
<body>

    <header>
        <div>
            <h1>Smart Daily Task Assistant</h1>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 4px;">
                Agentic AI Assistant Powered by Python and Ollama
            </p>
        </div>
        <div id="ai-status-badge" class="badge badge-fallback">Checking AI...</div>
    </header>

    <!-- Daily Summary Metrics -->
    <div class="summary-grid">
        <div class="summary-card">
            <div class="label">Total Tasks</div>
            <div class="val" id="stat-total" style="color: #60a5fa;">0</div>
        </div>
        <div class="summary-card">
            <div class="label">Pending</div>
            <div class="val" id="stat-pending" style="color: #fbbf24;">0</div>
        </div>
        <div class="summary-card">
            <div class="label">Completed</div>
            <div class="val" id="stat-completed" style="color: #34d399;">0</div>
        </div>
        <div class="summary-card">
            <div class="label">Urgent / Due Today</div>
            <div class="val" id="stat-urgent" style="color: #f87171;">0</div>
        </div>
    </div>

    <!-- AI Recommendation Banner -->
    <div class="ai-box" id="ai-recommendation-box" style="display: none;">
        <h3>AI Prioritization Recommendation</h3>
        <p id="ai-rec-text" style="font-size: 0.95rem; margin-bottom: 6px;"></p>
        <div id="ai-rec-reason" style="font-size: 0.85rem; color: var(--text-secondary);"></div>
    </div>

    <div class="layout-grid">
        <!-- Left Column: Task Board & Add Form -->
        <div>
            <!-- Add Task Form -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Add New Task</div>
                </div>
                <form id="add-task-form">
                    <div class="form-row">
                        <label>Task Name / Title</label>
                        <input type="text" id="task-title" placeholder="e.g. Finish Operating Systems Assignment" required>
                    </div>
                    <div class="form-cols">
                        <div class="form-row">
                            <label>Priority</label>
                            <select id="task-priority">
                                <option value="High">High</option>
                                <option value="Medium" selected>Medium</option>
                                <option value="Low">Low</option>
                            </select>
                        </div>
                        <div class="form-row">
                            <label>Deadline (Optional)</label>
                            <input type="date" id="task-deadline">
                        </div>
                    </div>
                    <button type="submit" class="btn-success" style="width: 100%;">+ Add Task</button>
                </form>
            </div>

            <!-- Pending Tasks -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Pending Tasks (<span id="count-pending">0</span>)</div>
                    <button onclick="fetchPrioritization()" class="btn-amber" style="padding: 6px 12px; font-size: 0.8rem;">
                        Analyze Priority
                    </button>
                </div>
                <div class="task-list" id="pending-list">
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">No pending tasks.</p>
                </div>
            </div>

            <!-- Completed Tasks -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Completed Tasks (<span id="count-completed">0</span>)</div>
                </div>
                <div class="task-list" id="completed-list">
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">No completed tasks yet.</p>
                </div>
            </div>
        </div>

        <!-- Right Column: Smart AI Agent Mode -->
        <div>
            <div class="panel" style="position: sticky; top: 20px;">
                <div class="panel-header">
                    <div class="panel-title">Smart AI Agent Mode</div>
                    <span style="font-size: 0.78rem; color: var(--text-secondary);">Intent -> Action -> Tool</span>
                </div>
                <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 12px;">
                    Speak in natural language! Try:
                    <br>&bull; <i>"Add my DBMS project submission due tomorrow with high priority"</i>
                    <br>&bull; <i>"Which task should I complete first?"</i>
                    <br>&bull; <i>"Mark task 1 as completed"</i>
                    <br>&bull; <i>"Give me my daily summary"</i>
                </p>

                <div class="chat-history" id="chat-history">
                    <div class="msg msg-agent">
                        Hello! I am your Smart Daily Task Assistant. Tell me what you'd like to do in plain English!
                    </div>
                </div>

                <form id="agent-form" style="display: flex; gap: 8px;">
                    <input type="text" id="agent-input" placeholder="Type a command in natural language..." required autocomplete="off">
                    <button type="submit">Send</button>
                </form>
            </div>
        </div>
    </div>

    <script>
        async function loadData() {
            try {
                const res = await fetch('/api/tasks');
                const data = await res.json();
                
                // Update AI badge
                const badge = document.getElementById('ai-status-badge');
                if (data.ollama_online) {
                    badge.className = 'badge badge-online';
                    badge.textContent = 'Ollama Online (' + data.model + ')';
                } else {
                    badge.className = 'badge badge-fallback';
                    badge.textContent = 'Rule Fallback Engine';
                }

                // Update summary stats
                document.getElementById('stat-total').textContent = data.summary.total_count;
                document.getElementById('stat-pending').textContent = data.summary.pending_count;
                document.getElementById('stat-completed').textContent = data.summary.completed_count;
                document.getElementById('stat-urgent').textContent = (data.summary.urgent_tasks || []).length;
                document.getElementById('count-pending').textContent = data.pending.length;
                document.getElementById('count-completed').textContent = data.completed.length;

                // Render Pending Tasks
                const pendingContainer = document.getElementById('pending-list');
                if (data.pending.length === 0) {
                    pendingContainer.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.9rem;">No pending tasks. You are all caught up!</p>';
                } else {
                    pendingContainer.innerHTML = data.pending.map(t => `
                        <div class="task-item">
                            <div>
                                <div class="task-title">#${t.id}: ${escapeHtml(t.title)}</div>
                                <div class="task-meta">
                                    <span class="p-${t.priority}">Priority: ${t.priority}</span>
                                    <span>Deadline: ${t.deadline || 'None'}</span>
                                </div>
                            </div>
                            <button class="btn-success" onclick="completeTask(${t.id}, '${escapeHtml(t.title)}')" style="padding: 6px 12px; font-size: 0.8rem;">
                                Complete
                            </button>
                        </div>
                    `).join('');
                }

                // Render Completed Tasks
                const completedContainer = document.getElementById('completed-list');
                if (data.completed.length === 0) {
                    completedContainer.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.9rem;">No completed tasks yet.</p>';
                } else {
                    completedContainer.innerHTML = data.completed.map(t => `
                        <div class="task-item completed">
                            <div>
                                <div class="task-title">#${t.id}: ${escapeHtml(t.title)}</div>
                                <div class="task-meta">
                                    <span>Completed on: ${t.completed_at || 'Done'}</span>
                                </div>
                            </div>
                            <span style="font-size: 0.8rem; color: #34d399; font-weight: 600;">[DONE]</span>
                        </div>
                    `).join('');
                }

            } catch (err) {
                console.error("Error loading task data:", err);
            }
        }

        async function completeTask(id, title) {
            if (!confirm(`Are you sure you want to mark Task #${id} ('${title}') as completed?`)) {
                return;
            }
            try {
                const res = await fetch(`/api/tasks/${id}/complete`, { method: 'POST' });
                const result = await res.json();
                if (result.success) {
                    loadData();
                } else {
                    alert(result.message);
                }
            } catch (err) {
                alert("Failed to complete task: " + err);
            }
        }

        async function fetchPrioritization() {
            try {
                const res = await fetch('/api/prioritize');
                const data = await res.json();
                const box = document.getElementById('ai-recommendation-box');
                const recText = document.getElementById('ai-rec-text');
                const recReason = document.getElementById('ai-rec-reason');

                if (data.recommended_task) {
                    box.style.display = 'block';
                    recText.innerHTML = `<strong>Complete First:</strong> #${data.recommended_task.id} - ${escapeHtml(data.recommended_task.title)} <span class="p-${data.recommended_task.priority}">(${data.recommended_task.priority})</span>`;
                    recReason.innerHTML = `<strong>Reasoning [${data.source}]:</strong> ${escapeHtml(data.reasoning)}`;
                    box.scrollIntoView({ behavior: 'smooth' });
                } else {
                    alert(data.reasoning || "No pending tasks to prioritize.");
                }
            } catch (err) {
                alert("Failed to prioritize tasks: " + err);
            }
        }

        document.getElementById('add-task-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const title = document.getElementById('task-title').value.trim();
            const priority = document.getElementById('task-priority').value;
            const deadline = document.getElementById('task-deadline').value || null;

            if (!title) return;

            try {
                const res = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: jsonStringify({ title, priority, deadline })
                });
                const added = await res.json();
                document.getElementById('task-title').value = '';
                document.getElementById('task-deadline').value = '';
                loadData();
            } catch (err) {
                alert("Failed to add task: " + err);
            }
        });

        // Smart Agent Form Submission
        document.getElementById('agent-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const input = document.getElementById('agent-input');
            const query = input.value.trim();
            if (!query) return;

            appendChat(query, 'user');
            input.value = '';

            try {
                const res = await fetch('/api/smart-agent', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: jsonStringify({ prompt: query })
                });
                const data = await res.json();
                appendAgentResponse(data);
                loadData();
            } catch (err) {
                appendChat("Error processing request: " + err, 'agent');
            }
        });

        function appendChat(text, sender) {
            const history = document.getElementById('chat-history');
            const div = document.createElement('div');
            div.className = `msg msg-${sender}`;
            div.textContent = text;
            history.appendChild(div);
            history.scrollTop = history.scrollHeight;
        }

        function appendAgentResponse(data) {
            const history = document.getElementById('chat-history');
            const div = document.createElement('div');
            div.className = 'msg msg-agent';
            
            let html = `<div>${escapeHtml(data.message || 'Done.')}</div>`;
            if (data.details) {
                html += `<div class="msg-step"><strong>Action:</strong> ${data.action} | <strong>Engine:</strong> ${data.engine}</div>`;
            }
            div.innerHTML = html;
            history.appendChild(div);
            history.scrollTop = history.scrollHeight;
        }

        function escapeHtml(text) {
            if (!text) return '';
            const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
            return String(text).replace(/[&<>"']/g, m => map[m]);
        }

        function jsonStringify(obj) { return JSON.stringify(obj); }

        // Initial load and auto refresh every 5 seconds
        loadData();
        setInterval(loadData, 5000);
    </script>
</body>
</html>
"""


class TaskAssistantHTTPHandler(BaseHTTPRequestHandler):
    """Handles REST API and UI requests."""

    def _send_json(self, data: dict, status_code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)

        if url.path == "/" or url.path == "/index.html":
            self._send_html(HTML_PAGE)
            return

        if url.path == "/api/tasks":
            all_tasks = task_mgr.get_all_tasks()
            pending = task_mgr.get_pending_tasks()
            completed = task_mgr.get_completed_tasks()
            summary = task_analyzer.get_daily_summary()
            ollama_status = is_ollama_available()

            self._send_json({
                "tasks": all_tasks,
                "pending": pending,
                "completed": completed,
                "summary": summary,
                "ollama_online": ollama_status,
                "model": OLLAMA_MODEL
            })
            return

        if url.path == "/api/prioritize":
            rec = task_analyzer.prioritize_tasks(use_ai=True)
            self._send_json(rec)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            req_data = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            req_data = {}

        # Add task endpoint
        if url.path == "/api/tasks":
            title = req_data.get("title", "").strip()
            priority = req_data.get("priority", "Medium")
            deadline = req_data.get("deadline")

            if not title:
                self._send_json({"error": "Task title is required"}, status_code=400)
                return

            new_task = task_mgr.add_task(title=title, priority=priority, deadline=deadline)
            self._send_json(new_task, status_code=201)
            return

        # Complete task endpoint: /api/tasks/<id>/complete
        complete_match = re.match(r"^/api/tasks/(\d+)/complete$", url.path)
        if complete_match:
            task_id = int(complete_match.group(1))
            success, msg, task = task_mgr.mark_task_completed(task_id)
            self._send_json({"success": success, "message": msg, "task": task})
            return

        # Smart Agent AI endpoint
        if url.path == "/api/smart-agent":
            user_prompt = req_data.get("prompt", "").strip()
            if not user_prompt:
                self._send_json({"message": "Please provide a command.", "details": False})
                return

            # Agentic Loop
            decision = choose_action(user_prompt)
            action = decision.get("action")
            params = decision.get("params") or {}
            engine = decision.get("engine", "Fallback")

            msg = ""
            if action == "add_task":
                title = params.get("title") or "New Task"
                priority = params.get("priority") or "Medium"
                deadline = params.get("deadline")
                created = task_mgr.add_task(title=title, priority=priority, deadline=deadline)
                msg = f"Task #{created['id']} ('{created['title']}') added with {created['priority']} priority."

            elif action == "view_tasks":
                pending = task_mgr.get_pending_tasks()
                msg = f"You have {len(pending)} pending tasks. Check the board on your left!"

            elif action == "prioritize_tasks":
                rec = task_analyzer.prioritize_tasks(use_ai=True)
                if rec.get("recommended_task"):
                    t = rec["recommended_task"]
                    msg = f"Recommended to do FIRST: #{t['id']} ('{t['title']}') - {rec.get('reasoning')}"
                else:
                    msg = rec.get("reasoning", "No pending tasks.")

            elif action == "complete_task":
                task_id = params.get("task_id")
                if task_id:
                    success, res_msg, _ = task_mgr.mark_task_completed(task_id)
                    msg = res_msg
                else:
                    msg = "Please specify which task ID you want to mark as completed."

            elif action == "daily_summary":
                summary = task_analyzer.get_daily_summary()
                msg = f"Summary: {summary['total_count']} Total, {summary['completed_count']} Completed, {summary['pending_count']} Pending, {len(summary['urgent_tasks'])} Urgent."

            elif action == "exit":
                msg = "Goodbye! Have a productive day."
            else:
                msg = "I didn't recognize that intent. Try asking to add, view, complete, or prioritize tasks."

            self._send_json({
                "message": msg,
                "action": action,
                "engine": engine,
                "details": True
            })
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Print web access log
        print(f"[Web Server] {self.address_string()} - {format % args}")


def run_server(port=PORT):
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, TaskAssistantHTTPHandler)
    print(f"\n=======================================================")
    print(f" Smart Daily Task Assistant - Web UI")
    print(f" Server running at: http://localhost:{port}")
    print(f" Press Ctrl+C to stop the server.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server(PORT)

