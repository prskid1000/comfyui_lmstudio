import json
from typing import Any, Dict, List, Optional, Tuple

import requests


class LMStudioChat:
    CATEGORY = "LM Studio"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "chat"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple[str, Dict[str, Any]]]]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
            },
            "optional": {
                "system": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "model": (
                    "STRING",
                    {"default": ""},
                ),
                "base_url": (
                    "STRING",
                    {"default": "http://localhost:1234/v1"},
                ),
                "api_key": (
                    "STRING",
                    {"default": ""},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 256, "min": 1, "max": 8096, "step": 8},
                ),
                "stream": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "stop": (
                    "STRING",
                    {"default": ""},
                ),
            },
        }

    def _build_messages(self, system: str, prompt: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _headers(self, api_key: Optional[str]) -> Dict[str, str]:
        token = api_key.strip() if api_key else "lm-studio"
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def _parse_stop(self, stop: str) -> Optional[List[str]]:
        text = stop.strip()
        if not text:
            return None
        if "," in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            return parts or None
        return [text]

    def _aggregate_stream(self, resp) -> str:
        text_parts = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                    choices = event.get("choices")
                    if isinstance(choices, list) and choices:
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content")
                        if isinstance(piece, str):
                            text_parts.append(piece)
                except Exception:
                    continue
        return "".join(text_parts)

    def chat(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 256,
        stream: bool = False,
        stop: str = "",
    ) -> Tuple[str]:
        try:
            url = base_url.rstrip("/") + "/chat/completions"
            payload: Dict[str, Any] = {
                "model": model or "",
                "messages": self._build_messages(system, prompt),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "stream": bool(stream),
            }

            stop_list = self._parse_stop(stop)
            if stop_list:
                payload["stop"] = stop_list

            if stream:
                resp = requests.post(
                    url,
                    headers=self._headers(api_key),
                    data=json.dumps(payload),
                    timeout=600,
                    stream=True,
                )
                resp.raise_for_status()
                text = self._aggregate_stream(resp)
                return (text,)

            resp = requests.post(
                url,
                headers=self._headers(api_key),
                data=json.dumps(payload),
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return (content,)
                if "error" in data:
                    err = data.get("error")
                    if isinstance(err, dict) and "message" in err:
                        return (f"Error from server: {err['message']}",)
                    return (f"Error from server: {err}",)

            return (f"Unexpected response: {data}",)
        except requests.HTTPError as http_err:
            try:
                err_json = resp.json()  # type: ignore[name-defined]
            except Exception:
                err_json = None
            if err_json:
                return (f"HTTP {resp.status_code}: {err_json}",)  # type: ignore[attr-defined]
            return (f"HTTP error: {http_err}",)
        except Exception as e:  # noqa: BLE001
            return (f"Request failed: {e}",)


