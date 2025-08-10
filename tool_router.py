import json
from typing import Any, Dict, List, Tuple


class LMSToolRouter:
    CATEGORY = "LM Studio"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("selected_tool_name", "selected_args_json")
    FUNCTION = "route"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tool_calls_json": (
                    "STRING",
                    {"multiline": True, "default": "[]"},
                ),
                "tools_json": (
                    "STRING",
                    {"multiline": True, "default": "[]"},
                ),
                "strategy": (
                    ["first", "by_name"],
                    {"default": "first"},
                ),
            },
            "optional": {
                "tool_name": (
                    "STRING",
                    {"default": ""},
                ),
            },
        }

    def route(
        self,
        tool_calls_json: str,
        tools_json: str,
        strategy: str = "first",
        tool_name: str = "",
    ) -> Tuple[str, str]:
        try:
            calls = json.loads(tool_calls_json or "[]")
            tools = json.loads(tools_json or "[]")
        except Exception as e:  # noqa: BLE001
            return ("", json.dumps({"error": f"Invalid JSON: {e}"}))

        # Normalize to OpenAI tool schema if function_call legacy present
        normalized_calls: List[Dict[str, Any]] = []
        if isinstance(calls, list):
            for c in calls:
                if isinstance(c, dict) and c.get("type") == "function" and "function" in c:
                    normalized_calls.append(c)
                elif isinstance(c, dict) and "function_call" in c:
                    normalized_calls.append({
                        "type": "function",
                        "function": c.get("function_call", {}),
                    })

        selected = None
        if strategy == "first" and normalized_calls:
            selected = normalized_calls[0]
        elif strategy == "by_name" and tool_name:
            for c in normalized_calls:
                fn = c.get("function", {})
                if isinstance(fn, dict) and fn.get("name") == tool_name:
                    selected = c
                    break

        if not selected:
            return ("", "{}")

        fn = selected.get("function", {})
        name = fn.get("name") or ""
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        return (name, json.dumps(args))


