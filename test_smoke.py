"""End-to-end smoke test for JAS roles, approval gates, chat and document export.

Run from the project directory with: python -m pytest -q
This uses a temporary SQLite database and mocks the external model API.
"""
import os
import tempfile

_TMP = tempfile.TemporaryDirectory(prefix="jas-smoke-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP.name}/jas-test.db"
os.environ["JAS_STORAGE_DIR"] = f"{_TMP.name}/storage"
os.environ["JAS_JWT_SECRET"] = "smoke-test-only-long-secret-not-for-production-0123456789"
os.environ["JAS_BOOTSTRAP_ADMIN_EMAIL"] = "admin@example.com"
os.environ["JAS_BOOTSTRAP_ADMIN_PASSWORD"] = "AdminPass!123456"
os.environ["JAS_BOOTSTRAP_ADMIN_NAME"] = "JAS Test Admin"
os.environ["JAS_LLM_PROVIDER"] = "compatible"
os.environ["JAS_LLM_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["JAS_LLM_API_KEY"] = "not-used-by-mocked-test"
os.environ["JAS_LLM_MODEL"] = "test-model"
os.environ["JAS_ALLOWED_EMAIL_DOMAINS"] = "example.com"

from fastapi.testclient import TestClient  # noqa: E402
from app import main as main_module  # noqa: E402


def test_smoke_role_gates_chat_and_exports(monkeypatch):
    monkeypatch.setattr(main_module, "generate_chat", fake_generate_chat)
    with TestClient(main_module.app) as client:
        assert client.get("/api/health").status_code == 200

        admin_res = client.post("/api/auth/login", json={"email": "admin@example.com", "password": "AdminPass!123456"})
        assert admin_res.status_code == 200
        admin_headers = {"Authorization": f"Bearer {admin_res.json()['access_token']}"}
        assert client.get("/api/admin/overview", headers=admin_headers).status_code == 200

        manager_res = client.post("/api/admin/managers", headers=admin_headers, json={
            "email": "manager@example.com", "full_name": "JAS Manager", "password": "ManagerPass!123456"
        })
        assert manager_res.status_code == 201
        manager_headers = {"Authorization": f"Bearer {client.post('/api/auth/login', json={'email':'manager@example.com','password':'ManagerPass!123456'}).json()['access_token']}"}
        assert client.post("/api/admin/managers", headers=manager_headers, json={
            "email": "othermanager@example.com", "full_name": "Other Manager", "password": "ManagerPass!123456"
        }).status_code == 403

        rejected_domain = client.post("/api/auth/register", json={
            "email": "student@outside.org", "full_name": "Outside User", "password": "StudentPass!123456"
        })
        assert rejected_domain.status_code == 403
        reg = client.post("/api/auth/register", json={
            "email": "student@example.com", "full_name": "JAS Student", "password": "StudentPass!123456"
        })
        assert reg.status_code == 201
        student_login = client.post("/api/auth/login", json={"email": "student@example.com", "password": "StudentPass!123456"})
        student_headers = {"Authorization": f"Bearer {student_login.json()['access_token']}"}
        assert client.get("/api/conversations", headers=student_headers).status_code == 403

        users = client.get("/api/manage/users", headers=manager_headers).json()
        student_id = next(person["id"] for person in users if person["email"] == "student@example.com")
        approved = client.post(f"/api/manage/users/{student_id}/decision", headers=manager_headers, json={"status": "approved"})
        assert approved.status_code == 200
        assert client.get("/api/conversations", headers=student_headers).status_code == 200

        uploaded = client.post("/api/uploads", headers=student_headers, files={
            "file": ("question.txt", b"Write a Python program to add two numbers.", "text/plain")
        })
        assert uploaded.status_code == 201
        upload_id = uploaded.json()["id"]
        chat = client.post("/api/chat", headers=student_headers, json={
            "message": "Help me solve this question paper", "mode": "code", "attachment_ids": [upload_id]
        })
        assert chat.status_code == 200
        assert "Mock JAS response" in chat.json()["message"]["content"]

        admin_uploads = client.get("/api/admin/uploads", headers=admin_headers)
        assert admin_uploads.status_code == 200
        assert any(item["filename"] == "question.txt" for item in admin_uploads.json()["uploads"])

        pdf = client.post("/api/exports/document", headers=student_headers, json={
            "title": "Smoke Test Assignment", "content": "# Introduction\n- First point\n```python\nprint('hello')\n```", "format": "pdf"
        })
        assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
        docx = client.post("/api/exports/document", headers=student_headers, json={
            "title": "Smoke Test Assignment", "content": "# Introduction\n- First point", "format": "docx"
        })
        assert docx.status_code == 200 and "wordprocessingml" in docx.headers["content-type"]


async def fake_generate_chat(messages, mode="general"):
    return "Mock JAS response for test mode: " + mode
