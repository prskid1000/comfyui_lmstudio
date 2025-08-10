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


class LMStudioToolLoop:
    CATEGORY = "LM Studio"
    OUTPUT_NODE = True
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "messages_json", "tool_traces_json", "raw_json", "parsed_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "images": ("IMAGE", {}),
                "image_paths_json": ("STRING", {"multiline": True, "default": ""}),
                "system": ("STRING", {"multiline": True, "default": ""}),
                "transcript": ("STRING", {"multiline": True, "default": ""}),
                "history_json": ("STRING", {"multiline": True, "default": ""}),
                "tools_json": ("STRING", {"multiline": True, "default": ""}),
                "tool_choice": ("STRING", {"default": "auto"}),
                "max_steps": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1}),
                "model": ("STRING", {"default": ""}),
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 32768, "step": 8}),
                "max_images": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "vision_mode": (["auto", "ignore", "force"], {"default": "auto"}),
                "json_mode": ("BOOLEAN", {"default": False}),
                "response_format_json": ("STRING", {"multiline": True, "default": ""}),
                "reasoning_effort": (["", "low", "medium", "high"], {"default": ""}),
                "max_output_tokens": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 1}),
                "parallel_tool_calls": ("BOOLEAN", {"default": True}),
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
            if hasattr(images, "cpu"):
                arr = images.cpu().numpy()
            else:
                arr = images
            if arr is None:
                return []
            b = int(arr.shape[0]) if len(arr.shape) == 4 else 1
            out: List[str] = []
            for i in range(min(b, max_images)):
                img_np = arr[i] if len(arr.shape) == 4 else arr
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

    def _image_paths_to_data_url_list(self, image_paths_json: str, max_images: int) -> List[str]:
        if not image_paths_json.strip() or Image is None:
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

    def _build_content_with_images(self, prompt: str, data_urls: List[str]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if prompt.strip():
            content.append({"type": "text", "text": prompt})
        for url in data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def _start_messages(
        self,
        system: str,
        prompt: str,
        transcript: str,
        images: Any,
        image_paths_json: str,
        max_images: int,
        vision_mode: str,
        history_json: str,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if history_json.strip():
            try:
                parsed = json.loads(history_json)
                if isinstance(parsed, list):
                    messages.extend(parsed)
            except Exception:
                pass
        if system.strip() and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system})

        merged_prompt = prompt
        if transcript.strip():
            merged_prompt = f"{merged_prompt}\n\n[Voice Transcript]\n{transcript}" if merged_prompt.strip() else transcript

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

    def _find_tool_def(self, tools: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if isinstance(fn, dict) and fn.get("name") == name:
                return t
        return None

    def _exec_http_tool(self, tool_def: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        http_cfg = tool_def.get("x-http") or {}
        if not isinstance(http_cfg, dict):
            # fallback legacy
            http_cfg = {
                "method": tool_def.get("x_method"),
                "url": tool_def.get("x_url"),
                "headers": tool_def.get("x_headers"),
                "body_template": tool_def.get("x_body_template"),
            }
        method = (http_cfg.get("method") or "GET").upper()
        url_tmpl = http_cfg.get("url") or ""
        headers = http_cfg.get("headers") or {}
        body_tmpl = http_cfg.get("body_template") or None

        def fmt(value: Any) -> Any:
            if isinstance(value, str):
                try:
                    return value.format(**args)
                except Exception:
                    return value
            if isinstance(value, dict):
                return {k: fmt(v) for k, v in value.items()}
            if isinstance(value, list):
                return [fmt(v) for v in value]
            return value

        url = fmt(url_tmpl)
        hdrs = fmt(headers) if isinstance(headers, dict) else {}
        body = fmt(body_tmpl) if isinstance(body_tmpl, (dict, list)) else None

        try:
            resp = requests.request(method=method, url=url, headers=hdrs, json=body, timeout=120)
            content_type = resp.headers.get("Content-Type", "")
            try:
                data = resp.json() if "application/json" in content_type else {"text": resp.text}
            except Exception:
                data = {"text": resp.text}
            return {
                "ok": resp.ok,
                "status": resp.status_code,
                "url": url,
                "data": data,
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def run(
        self,
        prompt: str,
        images: Any = None,
        image_paths_json: str = "",
        system: str = "",
        transcript: str = "",
        history_json: str = "",
        tools_json: str = "",
        tool_choice: str = "auto",
        max_steps: int = 3,
        model: str = "",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_images: int = 1,
        vision_mode: str = "auto",
        json_mode: bool = False,
        response_format_json: str = "",
        reasoning_effort: str = "",
        max_output_tokens: int = 0,
        parallel_tool_calls: bool = True,
    ) -> Tuple[str, str, str, str, str]:
        url = base_url.rstrip("/") + "/chat/completions"
        messages = self._start_messages(
            system=system,
            prompt=prompt,
            transcript=transcript,
            images=images,
            image_paths_json=image_paths_json,
            max_images=max_images,
            vision_mode=vision_mode,
            history_json=history_json,
        )

        try:
            tools: List[Dict[str, Any]] = json.loads(tools_json) if tools_json.strip() else []
        except Exception:
            tools = []

        traces: List[Dict[str, Any]] = []
        last_raw: Dict[str, Any] = {}

        for _ in range(max_steps + 1):
            payload: Dict[str, Any] = {
                "model": model or "",
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "stream": False,
            }
            if tools:
                payload["tools"] = tools
                if tool_choice:
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

            # Reasoning and limits
            if reasoning_effort:
                payload["reasoning"] = {"effort": reasoning_effort}
            if max_output_tokens and max_output_tokens > 0:
                payload["max_output_tokens"] = int(max_output_tokens)
            payload["parallel_tool_calls"] = bool(parallel_tool_calls)

            resp = requests.post(url, headers=self._headers(api_key), data=json.dumps(payload), timeout=300)
            resp.raise_for_status()
            data = resp.json()
            last_raw = data if isinstance(data, dict) else {"data": data}

            choices = last_raw.get("choices") if isinstance(last_raw, dict) else None
            if not (isinstance(choices, list) and choices):
                break
            message = choices[0].get("message") or {}
            content = message.get("content") if isinstance(message, dict) else None
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None

            if isinstance(tool_calls, list) and tool_calls:
                # Execute each tool call and append tool results to messages
                for tc in tool_calls:
                    fn = (tc or {}).get("function") or {}
                    name = fn.get("name") or ""
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"_raw": args}
                    tool_def = self._find_tool_def(tools, name) if tools else None
                    if tool_def is None:
                        result = {"ok": False, "error": f"Unknown tool: {name}", "args": args}
                        traces.append({"tool": name, "result": result})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", name),
                            "content": json.dumps(result),
                        })
                        continue
                    result = self._exec_http_tool(tool_def, args if isinstance(args, dict) else {})
                    traces.append({"tool": name, "args": args, "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", name),
                        "content": json.dumps(result),
                    })
                # Continue loop to let the model see tool outputs
                continue

            # No tool calls → final assistant output
            final_text = content if isinstance(content, str) else ""
            messages.append({"role": "assistant", "content": final_text})
            parsed = "null"
            if final_text and (json_mode or response_format_json.strip()):
                try:
                    parsed = json.dumps(json.loads(final_text))
                except Exception:
                    parsed = "null"
            return (
                final_text,
                json.dumps(messages),
                json.dumps(traces),
                json.dumps(last_raw),
                parsed,
            )

        # Exceeded steps or unexpected response
        return (
            "",
            json.dumps(messages),
            json.dumps(traces),
            json.dumps(last_raw),
            "null",
        )


