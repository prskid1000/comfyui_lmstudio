import json
from typing import Any, Dict, Optional, Tuple

import requests


class LMSToolHTTPExecutor:
    CATEGORY = "LM Studio"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result_text", "result_json")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tool_name": ("STRING", {"default": ""}),
                "args_json": ("STRING", {"multiline": True, "default": "{}"}),
                "method": (["GET", "POST", "PUT", "DELETE"], {"default": "GET"}),
                "url_template": ("STRING", {"default": ""}),
            },
            "optional": {
                "headers_json": ("STRING", {"multiline": True, "default": "{}"}),
                "body_json": ("STRING", {"multiline": True, "default": "{}"}),
                "timeout": ("INT", {"default": 60, "min": 1, "max": 600}),
            },
        }

    def _format_url(self, template: str, args: Dict[str, Any]) -> str:
        try:
            return template.format(**args)
        except Exception:
            return template

    def execute(
        self,
        tool_name: str,
        args_json: str,
        method: str,
        url_template: str,
        headers_json: str = "{}",
        body_json: str = "{}",
        timeout: int = 60,
    ) -> Tuple[str, str]:
        try:
            args = json.loads(args_json or "{}")
            headers = json.loads(headers_json or "{}")
            body = json.loads(body_json or "{}")
        except Exception as e:  # noqa: BLE001
            err = {"error": f"Invalid JSON: {e}"}
            return (json.dumps(err), json.dumps(err))

        url = self._format_url(url_template, args if isinstance(args, dict) else {})
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers if isinstance(headers, dict) else {},
                json=body if method in ("POST", "PUT", "PATCH") and isinstance(body, dict) else None,
                timeout=timeout,
            )
            content_type = resp.headers.get("Content-Type", "")
            try:
                data = resp.json() if "application/json" in content_type else {"text": resp.text}
            except Exception:
                data = {"text": resp.text}
            if resp.ok:
                result_text = data.get("text") if isinstance(data, dict) else str(data)
                if not result_text:
                    result_text = json.dumps(data)
                return (result_text, json.dumps({"status": resp.status_code, "data": data}))
            return (
                f"HTTP {resp.status_code}",
                json.dumps({"status": resp.status_code, "data": data}),
            )
        except Exception as e:  # noqa: BLE001
            err = {"error": str(e)}
            return (json.dumps(err), json.dumps(err))


