import base64
import hashlib
import os
from pathlib import Path
from urllib.parse import quote
import httpx

SYSTEM_PROMPTS = {
    "general": "You are JAS, a helpful campus AI assistant. Be accurate, practical, transparent about uncertainty, and clear about sources or assumptions. Never pretend to have performed actions you did not perform.",
    "study": "You are JAS Study Coach. Teach step by step, explain concepts in student-friendly language, use examples and short quizzes when helpful, and encourage understanding rather than copying without learning.",
    "code": "You are JAS Coding Assistant. Provide correct, runnable code in the requested language, state assumptions, explain how to run it, consider edge cases, and include tests where useful. Never claim code was executed unless an execution tool actually ran it.",
    "assignment": "You are JAS Academic Writing Assistant. Help structure a clear assignment with a title, headings, introduction, body, conclusion, and references only when credible sources are provided or known. Do not invent citations. Encourage the student to verify course requirements and submit work consistent with college policy.",
    "decision": "You are JAS Decision Support. Do not make consequential decisions autonomously. Clarify the objective, list plausible options, compare benefits, costs, risks, assumptions, and uncertainties, then recommend a next step with a human decision-maker in control.",
}


def _provider_config() -> tuple[str, str, str]:
    provider = os.getenv("JAS_LLM_PROVIDER", "compatible").strip().lower()
    model = os.getenv("JAS_LLM_MODEL", "gpt-4.1-mini").strip()
    if provider in {"compatible", "openai", "openai-compatible", "ollama", "local"}:
        provider = "compatible"
        api_key = os.getenv("JAS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    elif provider in {"google", "gemini"}:
        provider = "gemini"
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("JAS_LLM_API_KEY") or ""
    elif provider in {"anthropic", "claude"}:
        provider = "anthropic"
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("JAS_LLM_API_KEY") or ""
    else:
        raise RuntimeError("Unsupported JAS_LLM_PROVIDER. Use compatible, gemini, or anthropic.")
    if not api_key:
        raise RuntimeError(f"No API key configured for provider '{provider}'. Set its API key or JAS_LLM_API_KEY.")
    return provider, api_key, model


def _data_url_parts(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:") or "," not in url:
        return None
    header, data = url.split(",", 1)
    if ";base64" not in header:
        return None
    mime = header[5:].split(";", 1)[0] or "image/png"
    return mime, data


def _merge_adjacent_roles(messages: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if merged and merged[-1]["role"] == role:
            previous = merged[-1]["content"]
            if isinstance(previous, str) and isinstance(content, str):
                merged[-1]["content"] = previous + "\n\n" + content
            else:
                previous_parts = previous if isinstance(previous, list) else [{"type": "text", "text": str(previous)}]
                current_parts = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
                merged[-1]["content"] = previous_parts + current_parts
        else:
            merged.append({"role": role, "content": content})
    return merged


def _openai_to_gemini_parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict] = []
    for item in content or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append({"text": str(item.get("text", ""))})
        elif item.get("type") == "image_url":
            image_info = item.get("image_url") or {}
            parsed = _data_url_parts(str(image_info.get("url", "")))
            if parsed:
                mime, data = parsed
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
            elif image_info.get("url"):
                parts.append({"fileData": {"mimeType": "image/png", "fileUri": image_info["url"]}})
    return parts or [{"text": ""}]


def _openai_to_anthropic_content(content) -> list[dict] | str:
    if isinstance(content, str):
        return content
    blocks: list[dict] = []
    for item in content or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            blocks.append({"type": "text", "text": str(item.get("text", ""))})
        elif item.get("type") == "image_url":
            image_info = item.get("image_url") or {}
            parsed = _data_url_parts(str(image_info.get("url", "")))
            if parsed:
                mime, data = parsed
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
            elif image_info.get("url"):
                blocks.append({"type": "image", "source": {"type": "url", "url": image_info["url"]}})
    return blocks or [{"type": "text", "text": ""}]


async def _generate_compatible(messages: list[dict], mode: str, api_key: str, model: str) -> str:
    base_url = os.getenv("JAS_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = [{"role": "system", "content": SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])}] + messages
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": payload, "temperature": 0.3},
        )
        response.raise_for_status()
        data = response.json()
    choices = data.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content.strip() if isinstance(content, str) and content.strip() else "I could not produce a text response. Please try rephrasing your request."


async def _generate_gemini(messages: list[dict], mode: str, api_key: str, model: str) -> str:
    model = model.removeprefix("models/")
    endpoint = os.getenv("JAS_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    url = f"{endpoint}/models/{quote(model, safe='')}:generateContent"
    gemini_contents = []
    for message in _merge_adjacent_roles(messages):
        role = "model" if message["role"] == "assistant" else "user"
        gemini_contents.append({"role": role, "parts": _openai_to_gemini_parts(message.get("content", ""))})
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])}]},
        "contents": gemini_contents,
        "generationConfig": {"temperature": 0.3},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers={"x-goog-api-key": api_key}, json=body)
        response.raise_for_status()
        data = response.json()
    candidates = data.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    content = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text"))
    return content.strip() or "I could not produce a text response. Please try rephrasing your request."


async def _generate_anthropic(messages: list[dict], mode: str, api_key: str, model: str) -> str:
    endpoint = os.getenv("JAS_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/") + "/messages"
    claude_messages = [
        {"role": message["role"], "content": _openai_to_anthropic_content(message.get("content", ""))}
        for message in _merge_adjacent_roles(messages)
    ]
    body = {
        "model": model,
        "max_tokens": int(os.getenv("JAS_LLM_MAX_TOKENS", "4096")),
        "system": SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"]),
        "messages": claude_messages,
    }
    headers = {"x-api-key": api_key, "anthropic-version": os.getenv("JAS_ANTHROPIC_VERSION", "2023-06-01")}
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(endpoint, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
    blocks = data.get("content") or []
    content = "\n".join(block.get("text", "") for block in blocks if isinstance(block, dict) and block.get("type") == "text")
    return content.strip() or "I could not produce a text response. Please try rephrasing your request."


async def generate_chat(messages: list[dict], mode: str = "general") -> str:
    provider, api_key, model = _provider_config()
    if provider == "gemini":
        return await _generate_gemini(messages, mode, api_key, model)
    if provider == "anthropic":
        return await _generate_anthropic(messages, mode, api_key, model)
    return await _generate_compatible(messages, mode, api_key, model)


async def transcribe_media(path: Path) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Audio/video transcription is not configured. Set OPENAI_API_KEY to enable it.")
    endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/audio/transcriptions"
    model = os.getenv("JAS_TRANSCRIPTION_MODEL", "whisper-1")
    with path.open("rb") as media_file:
        files = {"file": (path.name, media_file, "application/octet-stream")}
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, data={"model": model}, files=files)
            response.raise_for_status()
            data = response.json()
    return str(data.get("text", "")).strip()


async def generate_image(prompt: str, user_identifier: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Image generation is not configured. Set OPENAI_API_KEY to enable it.")
    endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/images/generations"
    model = os.getenv("JAS_IMAGE_MODEL", "gpt-image-2.5-flare")
    payload = {"model": model, "prompt": prompt, "size": "1024x1024"}
    if user_identifier:
        # Stable privacy-preserving identifier; do not send the email address itself.
        payload["safety_identifier"] = hashlib.sha256(user_identifier.strip().lower().encode("utf-8")).hexdigest()[:64]
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    items = data.get("data") or []
    image_b64 = items[0].get("b64_json") if items else None
    if image_b64:
        return "data:image/png;base64," + image_b64
    image_url = items[0].get("url") if items else None
    if image_url:
        async with httpx.AsyncClient(timeout=60.0) as client:
            image_response = await client.get(image_url)
            image_response.raise_for_status()
        content_type = image_response.headers.get("content-type", "image/png")
        encoded = base64.b64encode(image_response.content).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
    raise RuntimeError("The image provider returned no image bytes or URL.")
