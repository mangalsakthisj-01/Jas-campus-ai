# JAS — Campus AI (starter release)

JAS is a deployable starter for a college-only AI workspace with user, manager, and admin roles. It includes approval-based access, chat, upload processing, admin upload review, audit logs, and PDF/Word export.

**Important architecture note:** this is an AI application, not a newly trained frontier language model. Gemini- or Claude-class capability depends on the model provider you connect. JAS uses an OpenAI-compatible Chat Completions endpoint, so you can configure a hosted model API or a local model server such as Ollama. The app does not train or fine-tune a model from student uploads.

## Included features

- Registration creates a pending student/user account; a manager or admin can approve or reject it.
- Only an admin can create manager accounts. Managers can manage student/user access, but cannot create admins or managers.
- Role-protected chat, conversation history, and file ownership checks.
- Upload PDF, DOCX, text/code files, images, audio, and common video formats (up to the configured upload limit).
- PDF and DOCX text extraction; current-message images are sent to a vision-capable model as image input.
- Optional audio transcription and video-audio extraction (requires `OPENAI_API_KEY` and `ffmpeg` in the container).
- Export an assistant response as a formatted PDF or Word document.
- Optional image generation (requires image API access and `OPENAI_API_KEY`).
- Admin dashboard with account counts, upload metadata and extracted text (configurable), file download, and audit events.
- PostgreSQL for Docker deployment; SQLite for simple local development.
- An optional Ollama service in Docker Compose for local model hosting.

## Quickest route: Docker Compose

Prerequisites: Docker Engine/Desktop with Compose v2.

1. Copy `.env.example` to `.env`.
2. Change `JAS_JWT_SECRET`, `JAS_BOOTSTRAP_ADMIN_PASSWORD`, and `POSTGRES_PASSWORD` to unique random values (e.g. hex output from `openssl rand -hex 32`). Use hex/alphanumeric passwords for the Compose database URL so punctuation does not need URL escaping. Do not deploy with example values.
3. Configure a model provider (see below). For hosted inference set `JAS_LLM_BASE_URL`, `JAS_LLM_API_KEY`, and `JAS_LLM_MODEL`. For separate image generation/transcription set `OPENAI_API_KEY` too. The example image model uses `gpt-image-2.5-flare`; availability and organization verification may depend on your provider account.
4. Start JAS:

   ```bash
   docker compose up --build -d
   docker compose logs -f app
   ```

5. Open `http://localhost:8000`. Sign in using the bootstrap admin credentials you placed in `.env`.
6. As admin, create manager accounts from **Control center**. Students register for an account and remain pending until an admin or manager approves them.

The bootstrap admin is created only when there is no approved admin account. After first setup, you may remove `JAS_BOOTSTRAP_ADMIN_PASSWORD` from the deployment configuration and store other secrets using a secrets manager. The bootstrap settings do not reset an existing account password.

### Hosted model endpoint

Example `.env` settings for OpenAI-compatible hosted chat inference:

```dotenv
JAS_LLM_PROVIDER=compatible
JAS_LLM_BASE_URL=https://api.openai.com/v1
JAS_LLM_API_KEY=your-model-provider-key
JAS_LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=your-openai-key-for-image-generation-and-transcription
JAS_IMAGE_MODEL=gpt-image-2.5-flare
JAS_TRANSCRIPTION_MODEL=whisper-1
```

For a direct **Gemini** connection, set `JAS_LLM_PROVIDER=gemini`, `GEMINI_API_KEY`, and `JAS_LLM_MODEL` to a model available on your Google AI account. The default endpoint is `https://generativelanguage.googleapis.com/v1beta`.

For a direct **Claude** connection, set `JAS_LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, and `JAS_LLM_MODEL` to a model available on your Anthropic account. The default endpoint is `https://api.anthropic.com/v1`.

For a local **Ollama** or other OpenAI-compatible endpoint, use `JAS_LLM_PROVIDER=compatible`, set `JAS_LLM_BASE_URL`, `JAS_LLM_API_KEY`, and `JAS_LLM_MODEL`. The model name must be available to that provider. Chat inference and optional image/transcription features may have separate access, limits, and charges. Do not commit `.env` to source control. For production secrets, use a secret manager instead of a committed dotenv file.

### Optional local model with Ollama

This is an alternative to hosted chat inference. A capable local model needs sufficient RAM/VRAM and the quality and speed depend on the selected model and hardware.

1. Enable the optional Ollama service:

   ```bash
   docker compose --profile localmodel up --build -d
   docker compose exec ollama ollama pull qwen3:8b
   ```

2. Set these values in `.env` and recreate the app:

   ```dotenv
   JAS_LLM_BASE_URL=http://ollama:11434/v1
   JAS_LLM_API_KEY=ollama
   JAS_LLM_MODEL=qwen3:8b
   ```

   ```bash
   docker compose up -d --force-recreate app
   ```

If you run JAS outside Docker, use the Ollama base URL exposed by your own machine (commonly `http://localhost:11434/v1`). Not every local model supports image input; use a vision-capable model for image questions. Image generation/transcription still use the separately configured OpenAI API in this starter.

## Run locally without Docker

Prerequisites: Python 3.11+ (Python 3.12 recommended). `ffmpeg` is optional and only needed to extract audio from video uploads.

```bash
cd jas
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

On Windows, copy `.env.example` to `.env` using File Explorer if `cp` is unavailable. Edit `.env` to set a strong `JAS_JWT_SECRET`, a bootstrap admin email/password, and your chosen model provider/API key. JAS automatically loads the project-local `.env` when the server starts. Then run:

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000`. For the simplest local run, SQLite is used automatically; Docker Compose is recommended for a shared deployment.

The local default database is `sqlite:///./data/jas.db`; uploads and generated files are stored under `./storage`. Keep these directories backed up and private.

## Roles and approval rules

| Capability | User / student | Manager | Admin |
|---|---:|---:|---:|
| Use chat and file tools | Approved account | Yes | Yes |
| Register an account | Yes (pending) | — | — |
| Approve/reject student accounts | No | Yes | Yes |
| Create manager accounts | No | No | Yes |
| See all upload metadata | No | No | Yes |
| Review extracted upload text / download another user's upload | No | No | Yes, if enabled/configured |
| View admin audit log and usage counts | No | No | Yes |

Managers can approve, reject, or disable normal user accounts. Managers cannot change admin/manager accounts. User accounts can download/delete only their own uploads. A user must be approved before using chat and upload endpoints.

## Upload visibility and student privacy

`JAS_ADMIN_CAN_VIEW_UPLOAD_CONTENT=true` allows admins to view extracted text and download uploaded files from the Control center. Set it to `false` to hide extracted text from the admin inbox (admins still see file metadata). Because student uploads can contain personal or sensitive data, decide this setting with the college, publish a clear notice/retention policy, limit who holds admin access, and collect consent where required. Set `JAS_MAX_UPLOAD_MB` to your institution's limit. Optionally set `JAS_ALLOWED_EMAIL_DOMAINS=college.edu,students.college.edu` to restrict new student registrations and manager creation to approved email domains; an empty value keeps approval-based access without domain filtering. Set `JAS_CORS_ORIGINS` only if the frontend is hosted on a separate trusted origin.

**Uploads do not update model weights.** Admin review is not model training. If the college later wants to improve a campus model, build an explicit opt-in dataset workflow with access controls, removal requests, de-identification, quality review, evaluation sets, and approval before any fine-tuning. Do not silently train on private student files.

## What file processing means in this starter

- **PDF:** extracts selectable text page by page. Scanned/image-only PDFs are not OCR'd yet; add a controlled OCR service (such as Tesseract) for those files.
- **DOCX:** extracts paragraphs and table cells. DOC legacy `.doc` is not currently enabled.
- **Images:** common image uploads are stored and attached to the model as image input for the current question. A vision-capable model is required.
- **Audio:** optional transcription uses the configured OpenAI transcription API.
- **Video:** optional audio track extraction uses `ffmpeg`; speech is then transcribed. This does not perform full video-frame analysis in the current version.
- **PDF/DOCX output:** export creates a basic document from the assistant response. It does not guarantee exact college template, citation correctness, or scanned-page fidelity; review before submission.

## API map

- `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/me`
- `GET/POST /api/conversations`, `GET /api/conversations/{id}/messages`, `POST /api/chat`
- `POST /api/uploads`, `GET /api/uploads`, `GET/DELETE /api/uploads/{id}`
- `POST /api/exports/document`, `POST /api/generate-image`
- `GET /api/manage/users`, `POST /api/manage/users/{id}/decision`
- `POST /api/admin/managers`, `GET /api/admin/overview`, `GET /api/admin/uploads`, `GET /api/admin/audit-logs`
- `GET /api/health`

Run the included role/upload/export smoke test with `pip install -r requirements-dev.txt` followed by `python -m pytest -q`. Interactive API docs are available at `/docs` during development. Do not expose the docs publicly in a hardened production configuration without reviewing whether that is appropriate.

## Production-readiness work before a campus-wide launch

This is a project starter, not a security-certified institutional platform. Before real deployment, add and test:

1. HTTPS behind a reverse proxy, secure cookie-based or institution-managed SSO authentication, MFA for admins, password reset, email/college-domain verification, and rate limiting.
2. A managed secret store, key rotation, database backups and restore drills, schema migrations, monitoring/alerts, and a retention/deletion policy.
3. Antivirus scanning, content-type and file-signature validation, stricter CPU/memory/time limits for document parsing, and isolated workers for untrusted media.
4. Per-user/per-college quotas, API cost budgets, abuse monitoring, output safety testing, prompt-injection defenses for retrieved files, and provider data-processing review.
5. A permission/privacy review. Audit records include actor, action, target and limited metadata; do not log secrets or full student chat content unnecessarily.
6. A formal evaluation set for coding, academic explanations, PDFs, images, Indian languages, and hallucination resistance. A bigger model alone does not guarantee correctness.

The app has role checks and private upload ownership checks, but it has not undergone an external penetration test. Treat it as a foundation for a college project and validate it before handling real student data.

## References

- OpenAI Responses API overview and multimodal/tool capabilities: https://developers.openai.com/api/reference/responses/overview
- OpenAI file inputs: https://developers.openai.com/api/docs/guides/file-inputs
- Gemini `generateContent` API: https://ai.google.dev/api/generate-content
- Anthropic Messages API / image and PDF guidance: https://docs.anthropic.com/en/api/messages and https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
- Docker Compose environment variables: https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/
- Docker Compose secrets: https://docs.docker.com/compose/how-tos/use-secrets/
