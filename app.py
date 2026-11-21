"""
Interactive Web Application for Anvesh - Agentic RAG Playground.
Features an interactive Benchmark Anchor Date editor, live CSV data manager
(for editing Students, Courses, Attendance, Assessments), and conversational reasoning.
"""

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.agent import SkillBridgeAgent
from src.config import REFERENCE_DATE_STR
from src.tools import lookup_student_data, save_dataframe, invalidate_csv_cache

logger = logging.getLogger("rag_web_app")
agent_instance = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anvesh - Agentic RAG Playground</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); padding: 24px 16px; min-height: 100vh; }
        .container { max-width: 950px; margin: 0 auto; }
        
        header { margin-bottom: 24px; border-bottom: 1px solid var(--card-border); padding-bottom: 16px; display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; }
        h1 { font-size: 24px; color: var(--accent); display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
        
        /* Anchor Date & Header Controls */
        .header-actions { display: flex; align-items: center; gap: 12px; }
        .badge { background: #0369a1; color: #e0f2fe; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
        .manage-btn { background: #334155; border: 1px solid #475569; color: #38bdf8; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .manage-btn:hover { background: #475569; border-color: var(--accent); color: white; }

        .benchmark-info { font-size: 13px; color: var(--text-muted); display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 6px; }
        .date-picker-input { background: #0f172a; border: 1px solid #38bdf8; color: #38bdf8; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 13px; cursor: pointer; outline: none; }
        .date-picker-input:focus { border-color: #7dd3fc; box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2); }
        .small-btn { background: #1e293b; border: 1px solid #475569; color: #cbd5e1; font-size: 11px; padding: 4px 8px; border-radius: 4px; cursor: pointer; }
        .small-btn:hover { background: #334155; color: white; }
        .date-badge { background: #1e3a8a; color: #93c5fd; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
        
        /* Cards */
        .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .card-title { font-size: 15px; font-weight: 600; margin-bottom: 12px; color: #cbd5e1; }
        
        .samples { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
        .sample-btn { background: #334155; border: 1px solid #475569; color: #e2e8f0; padding: 6px 12px; border-radius: 8px; font-size: 12px; cursor: pointer; transition: all 0.2s; }
        .sample-btn:hover { background: #475569; border-color: var(--accent); }
        
        .input-group { display: flex; gap: 12px; }
        input[type="text"] { flex: 1; background: #0f172a; border: 1px solid var(--card-border); border-radius: 8px; padding: 12px 16px; color: white; font-size: 14px; outline: none; }
        input[type="text"]:focus { border-color: var(--accent); }
        button#ask-btn { background: var(--accent); border: none; border-radius: 8px; padding: 12px 24px; color: #0f172a; font-weight: 700; cursor: pointer; transition: background 0.2s; }
        button#ask-btn:hover { background: var(--accent-hover); }

        .result-box { display: none; margin-top: 20px; border-top: 1px solid var(--card-border); padding-top: 20px; }
        .meta-row { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
        .meta-item { background: #0f172a; padding: 8px 14px; border-radius: 8px; font-size: 13px; border: 1px solid var(--card-border); }
        .meta-label { color: var(--text-muted); font-size: 11px; text-transform: uppercase; font-weight: bold; }
        .meta-val { font-weight: 600; font-size: 14px; margin-top: 2px; }

        .answer-display { background: #0f172a; border-left: 4px solid var(--accent); padding: 16px; border-radius: 6px; margin-bottom: 16px; }
        .answer-label { font-size: 12px; color: var(--text-muted); margin-bottom: 4px; text-transform: uppercase; font-weight: bold; }
        .answer-text { font-size: 18px; font-weight: 700; color: #38bdf8; word-break: break-word; }
        
        .reasoning-box { background: #0f172a; padding: 14px; border-radius: 8px; font-size: 13px; line-height: 1.5; color: #cbd5e1; border: 1px solid var(--card-border); }
        .loading { display: none; text-align: center; color: var(--accent); font-size: 14px; padding: 20px 0; }

        /* Modal for Editing Records */
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.75); z-index: 1000; align-items: center; justify-content: center; }
        .modal { background: #1e293b; border: 1px solid #475569; width: 90%; max-width: 900px; max-height: 85vh; border-radius: 12px; padding: 24px; display: flex; flex-direction: column; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        .modal-header h2 { font-size: 18px; color: var(--accent); }
        .close-btn { background: none; border: none; color: #94a3b8; font-size: 22px; cursor: pointer; }
        .close-btn:hover { color: white; }

        .tabs { display: flex; gap: 8px; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 8px; }
        .tab-btn { background: #0f172a; border: 1px solid #334155; color: #cbd5e1; padding: 6px 16px; border-radius: 6px; font-size: 13px; cursor: pointer; font-weight: 600; }
        .tab-btn.active { background: #0284c7; color: white; border-color: #38bdf8; }

        .table-container { flex: 1; overflow-y: auto; margin-bottom: 16px; border: 1px solid #334155; border-radius: 6px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { background: #0f172a; color: #94a3b8; text-align: left; padding: 8px 12px; border-bottom: 1px solid #334155; font-weight: 600; position: sticky; top: 0; }
        td { padding: 6px 12px; border-bottom: 1px solid #334155; color: #e2e8f0; }
        td input { background: transparent; border: 1px solid transparent; color: white; padding: 4px 6px; width: 100%; font-size: 13px; border-radius: 4px; }
        td input:focus { background: #0f172a; border-color: #38bdf8; outline: none; }
        
        .modal-footer { display: flex; justify-content: space-between; align-items: center; }
        .action-group { display: flex; gap: 10px; }
        .btn-add { background: #334155; border: 1px solid #475569; color: #e2e8f0; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }
        .btn-add:hover { background: #475569; }
        .btn-save { background: #22c55e; border: none; color: #0f172a; padding: 8px 20px; border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 13px; }
        .btn-save:hover { background: #16a34a; }
        .toast { display: none; color: #22c55e; font-size: 13px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🎓 Anvesh</h1>
                <div class="benchmark-info">
                    <span>Benchmark Anchor Date:</span>
                    <input type="date" id="anchor-date" value="2025-01-15" class="date-picker-input" onchange="onDateChanged(this.value)">
                    <button class="small-btn" onclick="resetAnchorDate()" title="Reset to default competition benchmark (2025-01-15)">Reset</button>
                    <span id="date-badge" class="date-badge">Active: 2025-01-15</span>
                </div>
            </div>
            <div class="header-actions">
                <button class="manage-btn" onclick="openDataModal()">✏️ Edit Data & Records</button>
                <span class="badge">Agentic RAG Engine</span>
            </div>
        </header>

        <div class="card">
            <div class="card-title">Quick Preset Queries</div>
            <div class="samples">
                <button class="sample-btn" onclick="setQuery('What is the weighted attendance percentage for student STU001 in course CS101?')">📊 Weighted Attendance (CS101)</button>
                <button class="sample-btn" onclick="setQuery('How many full weeks has student STU001 been enrolled as of the reference date?')">📅 Elapsed Weeks (STU001)</button>
                <button class="sample-btn" onclick="setQuery('What is the expected progress percentage for student STU001?')">📈 Expected Progress %</button>
                <button class="sample-btn" onclick="setQuery('What is the composite final score for student STU001 in course CS101?')">📝 Composite Score</button>
                <button class="sample-btn" onclick="setQuery('What letter grade does student STU001 receive in course CS101?')">🏆 Letter Grade</button>
                <button class="sample-btn" onclick="setQuery('Who is STU001?')">🔍 Student Profile</button>
                <button class="sample-btn" onclick="setQuery('List all courses')">📚 Course Catalog</button>
                <button class="sample-btn" onclick="setQuery('What is the outstanding tuition balance for student STU001?')">🚫 Out-of-Scope (Tuition)</button>
            </div>

            <div class="input-group">
                <input type="text" id="query-input" placeholder="Enter student performance, attendance, or policy query..." onkeypress="handleKey(event)">
                <button id="ask-btn" onclick="submitQuery()">Ask Agent</button>
            </div>

            <div class="loading" id="loading">⚡ Reasoning over policy documents and records...</div>

            <div class="result-box" id="result-box">
                <div class="meta-row">
                    <div class="meta-item">
                        <div class="meta-label">Category</div>
                        <div class="meta-val" id="res-category">-</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Answerable Flag</div>
                        <div class="meta-val" id="res-answerable">-</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Evaluation Date</div>
                        <div class="meta-val" id="res-eval-date">-</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Tool Calls</div>
                        <div class="meta-val" id="res-tools">-</div>
                    </div>
                </div>

                <div class="answer-display">
                    <div class="answer-label">Calculated Answer</div>
                    <div class="answer-text" id="res-answer">-</div>
                </div>

                <div class="reasoning-box">
                    <div class="meta-label" style="margin-bottom: 6px;">Reasoning & Rule Application</div>
                    <div id="res-reasoning">-</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Data Management Modal -->
    <div class="modal-overlay" id="data-modal">
        <div class="modal">
            <div class="modal-header">
                <h2>✏️ Edit Data & Records (Students, Courses, Attendance, Assessments)</h2>
                <button class="close-btn" onclick="closeDataModal()">&times;</button>
            </div>
            
            <div class="tabs">
                <button class="tab-btn active" id="tab-students" onclick="switchTable('students')">Students</button>
                <button class="tab-btn" id="tab-courses" onclick="switchTable('courses')">Courses</button>
                <button class="tab-btn" id="tab-attendance" onclick="switchTable('attendance')">Attendance</button>
                <button class="tab-btn" id="tab-assessments" onclick="switchTable('assessments')">Assessments</button>
            </div>

            <div class="table-container" id="table-container">
                <table id="records-table">
                    <thead><tr id="table-head"></tr></thead>
                    <tbody id="table-body"></tbody>
                </table>
            </div>

            <div class="modal-footer">
                <span class="toast" id="save-toast">✅ Changes saved! Agent memory updated.</span>
                <div class="action-group">
                    <button class="btn-add" onclick="addNewRow()">+ Add Row</button>
                    <button class="btn-save" onclick="saveTableChanges()">💾 Save Changes to CSV</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentActiveDate = document.getElementById('anchor-date').value;
        let currentTable = 'students';
        let currentTableData = [];
        let currentColumns = [];

        function onDateChanged(newDate) {
            currentActiveDate = newDate;
            document.getElementById('date-badge').textContent = 'Active: ' + newDate;
        }

        function resetAnchorDate() {
            document.getElementById('anchor-date').value = '2025-01-15';
            onDateChanged('2025-01-15');
        }

        function setQuery(text) {
            document.getElementById('query-input').value = text;
            submitQuery();
        }

        function handleKey(e) {
            if (e.key === 'Enter') submitQuery();
        }

        async function submitQuery() {
            const input = document.getElementById('query-input');
            const query = input.value.trim();
            if (!query) return;

            const loading = document.getElementById('loading');
            const resultBox = document.getElementById('result-box');
            loading.style.display = 'block';
            resultBox.style.display = 'none';

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        reference_date: currentActiveDate
                    })
                });
                const data = await res.json();
                
                document.getElementById('res-category').textContent = data.category || 'Unknown';
                
                const ansBadge = document.getElementById('res-answerable');
                ansBadge.textContent = data.answerable === 1 ? '1 (Answerable)' : '0 (Out-of-Scope)';
                ansBadge.style.color = data.answerable === 1 ? '#22c55e' : '#ef4444';

                document.getElementById('res-eval-date').textContent = currentActiveDate;
                document.getElementById('res-tools').textContent = data.tool_calls_count ?? 0;
                document.getElementById('res-answer').textContent = data.answer;
                document.getElementById('res-reasoning').textContent = data.reasoning;

                resultBox.style.display = 'block';
            } catch (err) {
                alert('Request failed: ' + err.message);
            } finally {
                loading.style.display = 'none';
            }
        }

        // Data Modal Functions
        function openDataModal() {
            document.getElementById('data-modal').style.display = 'flex';
            loadTable(currentTable);
        }

        function closeDataModal() {
            document.getElementById('data-modal').style.display = 'none';
        }

        function switchTable(tableName) {
            currentTable = tableName;
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('tab-' + tableName).classList.add('active');
            loadTable(tableName);
        }

        async function loadTable(tableName) {
            try {
                const res = await fetch('/api/data?table=' + tableName);
                const json = await res.json();
                currentColumns = json.columns || [];
                currentTableData = json.data || [];
                renderTable();
            } catch (err) {
                alert('Failed to load table ' + tableName + ': ' + err.message);
            }
        }

        function renderTable() {
            const thead = document.getElementById('table-head');
            const tbody = document.getElementById('table-body');
            thead.innerHTML = '';
            tbody.innerHTML = '';

            currentColumns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                thead.appendChild(th);
            });
            const actionTh = document.createElement('th');
            actionTh.textContent = 'Action';
            thead.appendChild(actionTh);

            currentTableData.forEach((row, rowIdx) => {
                const tr = document.createElement('tr');
                currentColumns.forEach(col => {
                    const td = document.createElement('td');
                    const input = document.createElement('input');
                    input.value = row[col] !== undefined ? row[col] : '';
                    input.onchange = (e) => {
                        currentTableData[rowIdx][col] = e.target.value;
                    };
                    td.appendChild(input);
                    tr.appendChild(td);
                });

                const delTd = document.createElement('td');
                const delBtn = document.createElement('button');
                delBtn.textContent = '🗑️';
                delBtn.className = 'small-btn';
                delBtn.onclick = () => {
                    currentTableData.splice(rowIdx, 1);
                    renderTable();
                };
                delTd.appendChild(delBtn);
                tr.appendChild(delTd);

                tbody.appendChild(tr);
            });
        }

        function addNewRow() {
            const newObj = {};
            currentColumns.forEach(col => newObj[col] = '');
            currentTableData.push(newObj);
            renderTable();
        }

        async function saveTableChanges() {
            try {
                const res = await fetch('/api/data/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        table: currentTable,
                        records: currentTableData
                    })
                });
                const result = await res.json();
                if (result.status === 'success') {
                    const toast = document.getElementById('save-toast');
                    toast.style.display = 'inline';
                    setTimeout(() => { toast.style.display = 'none'; }, 3000);
                } else {
                    alert('Error saving: ' + result.error);
                }
            } catch (err) {
                alert('Save failed: ' + err.message);
            }
        }
    </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif url.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "reference_date": REFERENCE_DATE_STR}).encode("utf-8"))
        elif url.path == "/api/data":
            query_params = parse_qs(url.query)
            table_name = query_params.get("table", ["students"])[0]
            try:
                res = lookup_student_data(table_name)
                data = res.get("data", [])
                cols = list(data[0].keys()) if data else []
                payload = {"status": "success", "table": table_name, "columns": cols, "data": data}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/query":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                query = data.get("query", "")
                reference_date = data.get("reference_date", REFERENCE_DATE_STR)

                global agent_instance
                if agent_instance is None:
                    agent_instance = SkillBridgeAgent()

                response = agent_instance.answer_query(query, reference_date=reference_date)
                res_payload = response.dict()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res_payload).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif url.path == "/api/data/update":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                table_name = data.get("table", "")
                records = data.get("records", [])

                save_dataframe(table_name, records)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": f"Updated {table_name}"}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress verbose standard HTTP server request logs
        return


def start_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Anvesh Agent Web Server running at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port)
