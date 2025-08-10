import base64
import io
import json
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


class LMStudioChatAdvanced:
    CATEGORY = "LM Studio"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "tool_calls_json", "raw_json", "messages_json", "parsed_json")
    FUNCTION = "chat"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
            },
            "optional": {
                "images": ("IMAGE", {}),
                "image_paths_json": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "system": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "transcript": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "history_json": (
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
                    {"default": 512, "min": 1, "max": 32768, "step": 8},
                ),
                "stream": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "stop": (
                    "STRING",
                    {"default": ""},
                ),
                "tools_json": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "tool_choice": (
                    "STRING",
                    {"default": ""},
                ),
                "max_images": (
                    "INT",
                    {"default": 1, "min": 0, "max": 16, "step": 1},
                ),
                "vision_mode": (
                    ["auto", "ignore", "force"],
                    {"default": "auto"},
                ),
                "append_assistant_to_history": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "json_mode": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "response_format_json": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "reasoning_effort": (
                    ["", "low", "medium", "high"],
                    {"default": ""},
                ),
                "max_output_tokens": (
                    "INT",
                    {"default": 0, "min": 0, "max": 32768, "step": 1},
                ),
                "parallel_tool_calls": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    # ---------- helpers ----------
    def _headers(self, api_key: Optional[str]) -> Dict[str, str]:
        token = (api_key or "lm-studio").strip() or "lm-studio"
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

    def _image_tensor_to_data_url_list(self, images: Any, max_images: int) -> List[str]:
        if images is None:
            return []
        if Image is None or np is None:
            return []
        try:
            # ComfyUI IMAGE: torch.Tensor [B,H,W,C] float in [0,1]
            # Avoid importing torch directly; rely on duck-typing via numpy conversion.
            batch = getattr(images, "shape", None)
            if batch is None:
                return []
            # Convert to numpy
            if hasattr(images, "cpu"):
                arr = images.cpu().numpy()
            else:
                arr = images
            if arr is None:
                return []
            b = int(arr.shape[0]) if len(arr.shape) == 4 else 1
            out: List[str] = []
            for i in range(min(b, max_images)):
                if len(arr.shape) == 4:
                    img_np = arr[i]
                else:
                    img_np = arr
                # Ensure HWC, uint8
                img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
                if img_np.shape[-1] == 1:
                    img_np = img_np[..., 0]
                pil_img = Image.fromarray(img_np)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                out.append(f"data:image/png;base64,{b64}")
            return out
        except Exception:
            return []

    def _build_content_with_images(self, prompt: str, data_urls: List[str]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        # Place text first for better instruction-following
        if prompt.strip():
            content.append({"type": "text", "text": prompt})
        for url in data_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        return content

    def _image_paths_to_data_url_list(self, image_paths_json: str, max_images: int) -> List[str]:
        if not image_paths_json.strip():
            return []
        if Image is None:
            return []
        try:
            paths = json.loads(image_paths_json)
            if not isinstance(paths, list):
                return []
            out: List[str] = []
            for p in paths[: max(0, max_images)]:
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGBA") if im.mode in ("LA", "P") else im.convert("RGB")
                        buf = io.BytesIO()
                        im.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        out.append(f"data:image/png;base64,{b64}")
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def _build_messages(
        self,
        system: str,
        prompt: str,
        transcript: str,
        images: Any,
        image_paths_json: str,
        max_images: int,
        vision_mode: str,
        base_history: Optional[List[Dict[str, Any]]] = None,
        history_json: str = "",
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        # Start from provided history if given
        if base_history is not None:
            messages.extend(base_history)
        elif history_json.strip():
            try:
                parsed = json.loads(history_json)
                if isinstance(parsed, list):
                    messages.extend(parsed)
            except Exception:
                pass
        # Prepend system if not already present
        if system.strip() and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system})

        # Merge transcript with prompt if provided
        merged_prompt = prompt
        if transcript.strip():
            if merged_prompt.strip():
                merged_prompt = f"{merged_prompt}\n\n[Voice Transcript]\n{transcript}"
            else:
                merged_prompt = transcript

        data_urls: List[str] = []
        if vision_mode != "ignore":
            data_urls = self._image_tensor_to_data_url_list(images, max_images)
            if not data_urls:
                data_urls = self._image_paths_to_data_url_list(image_paths_json, max_images)

        if data_urls and vision_mode in ("auto", "force"):
            content = self._build_content_with_images(merged_prompt, data_urls)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": merged_prompt})

        return messages

    def _extract_tool_calls(self, message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return tool_calls
        # Legacy function_call
        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            return [{
                "id": "function_call",
                "type": "function",
                "function": function_call,
            }]
        return None

    def _aggregate_stream(self, resp: requests.Response) -> Tuple[str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        text_parts: List[str] = []
        tool_calls_accum: List[Dict[str, Any]] = []
        raw_events: List[Dict[str, Any]] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                    raw_events.append(event)
                    choices = event.get("choices")
                    if isinstance(choices, list) and choices:
                        delta = choices[0].get("delta") or {}
                        # text deltas
                        content_piece = delta.get("content")
                        if isinstance(content_piece, str):
                            text_parts.append(content_piece)
                        # tool call deltas (OpenAI streams function/tool deltas)
                        tool_calls = delta.get("tool_calls")
                        if isinstance(tool_calls, list) and tool_calls:
                            tool_calls_accum = tool_calls  # best-effort
                except Exception:
                    continue
        text = "".join(text_parts)
        raw = {"events": raw_events}
        return text, (tool_calls_accum or None), raw

    # ---------- main ----------
    def chat(
        self,
        prompt: str,
        images: Any = None,
        image_paths_json: str = "",
        system: str = "",
        transcript: str = "",
        history_json: str = "",
        model: str = "",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        stop: str = "",
        tools_json: str = "",
        tool_choice: str = "",
        max_images: int = 1,
        vision_mode: str = "auto",
        append_assistant_to_history: bool = True,
        json_mode: bool = False,
        response_format_json: str = "",
        reasoning_effort: str = "",
        max_output_tokens: int = 0,
        parallel_tool_calls: bool = True,
    ) -> Tuple[str, str, str, str, str]:
        url = base_url.rstrip("/") + "/chat/completions"
        messages = self._build_messages(
            system=system,
            prompt=prompt,
            transcript=transcript,
            images=images,
            image_paths_json=image_paths_json,
            max_images=max_images,
            vision_mode=vision_mode,
            history_json=history_json,
        )

        payload: Dict[str, Any] = {
            "model": model or "",
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
        }

        stop_list = self._parse_stop(stop)
        if stop_list:
            payload["stop"] = stop_list

        if tools_json.strip():
            try:
                payload["tools"] = json.loads(tools_json)
            except Exception as e:  # noqa: BLE001
                err = f"Invalid tools_json: {e}"
                return (err, "[]", json.dumps({"error": err}))
        if tool_choice.strip():
            # OpenAI supports: "none" | "auto" | {"type":"function","function":{"name":"..."}}
            try:
                payload["tool_choice"] = json.loads(tool_choice)
            except Exception:
                payload["tool_choice"] = tool_choice

        # Structured output
        if response_format_json.strip():
            try:
                payload["response_format"] = json.loads(response_format_json)
            except Exception:
                pass
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}

        # Reasoning support (for compatible models)
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}

        if max_output_tokens and max_output_tokens > 0:
            payload["max_output_tokens"] = int(max_output_tokens)

        if isinstance(parallel_tool_calls, bool):
            payload["parallel_tool_calls"] = parallel_tool_calls

        try:
            if stream:
                resp = requests.post(
                    url,
                    headers=self._headers(api_key),
                    data=json.dumps(payload),
                    timeout=600,
                    stream=True,
                )
                resp.raise_for_status()
                text, tool_calls, raw = self._aggregate_stream(resp)
                tool_calls_json = json.dumps(tool_calls or [])
                raw_json = json.dumps(raw)
                # Append assistant message to history if requested
                out_history = messages[:]
                if append_assistant_to_history:
                    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": text}
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    out_history.append(assistant_msg)
                # Best-effort parse JSON from text when structured output was requested
                parsed_json = "null"
                if json_mode or response_format_json.strip():
                    try:
                        parsed_json = json.dumps(json.loads(text))
                    except Exception:
                        parsed_json = "null"
                return (text, tool_calls_json, raw_json, json.dumps(out_history), parsed_json)

            # non-stream
            resp = requests.post(
                url, headers=self._headers(api_key), data=json.dumps(payload), timeout=300
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    text = content if isinstance(content, str) else ""
                    tool_calls = self._extract_tool_calls(message) or []
                    out_history = messages[:]
                    if append_assistant_to_history:
                        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": text}
                        if tool_calls:
                            assistant_msg["tool_calls"] = tool_calls
                        out_history.append(assistant_msg)
                    return (
                        text,
                        json.dumps(tool_calls),
                        json.dumps(data),
                        json.dumps(out_history),
                        (json.dumps(json.loads(text)) if (json_mode or response_format_json.strip()) else "null") if text else "null",
                    )
                if "error" in data:
                    return (
                        f"Error: {data.get('error')}",
                        "[]",
                        json.dumps(data),
                        json.dumps(messages),
                        "null",
                    )
            return (f"Unexpected response: {data}", "[]", json.dumps(data), json.dumps(messages), "null")

        except requests.HTTPError as http_err:
            try:
                err_json = resp.json()  # type: ignore[name-defined]
            except Exception:
                err_json = None
            if err_json:
                return (
                    f"HTTP {resp.status_code}: {err_json}",  # type: ignore[attr-defined]
                    "[]",
                    json.dumps(err_json),
                    json.dumps(messages),
                    "null",
                )
            return (f"HTTP error: {http_err}", "[]", json.dumps({"error": str(http_err)}), json.dumps(messages), "null")
        except Exception as e:  # noqa: BLE001
            return (f"Request failed: {e}", "[]", json.dumps({"error": str(e)}), json.dumps(messages), "null")


