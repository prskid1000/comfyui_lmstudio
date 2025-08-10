import os
import json
from typing import Tuple


class LMStudioSaveText:
    CATEGORY = "LM Studio"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "format": (["auto", "txt", "json"], {"default": "auto"}),
                "pretty": ("BOOLEAN", {"default": True}),
                "filename": ("STRING", {"default": ""}),
                "directory": ("STRING", {"default": "output"}),
            },
        }

    def _detect_format(self, content: str, fmt: str) -> str:
        if fmt in ("txt", "json"):
            return fmt
        # auto
        try:
            json.loads(content)
            return "json"
        except Exception:
            return "txt"

    def _ensure_extension(self, filename: str, fmt: str) -> str:
        if not filename.strip():
            return f"lmstudio_output.{fmt}"
        name = filename
        ext = ".json" if fmt == "json" else ".txt"
        if not name.lower().endswith(ext):
            name += ext
        return name

    def save(
        self,
        content: str,
        format: str = "auto",
        pretty: bool = True,
        filename: str = "",
        directory: str = "output",
    ) -> Tuple[str]:
        try:
            fmt = self._detect_format(content, format)
            safe_dir = os.path.normpath(directory)
            os.makedirs(safe_dir, exist_ok=True)
            name = self._ensure_extension(filename, fmt)
            path = os.path.join(safe_dir, name)

            if fmt == "json":
                try:
                    data = json.loads(content)
                except Exception:
                    # Not valid JSON; save as raw string but with .json
                    data = {"text": content}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=(2 if pretty else None))
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

            print(f"[LMStudioSaveText] saved to {path}")
            return (path,)
        except Exception as e:  # noqa: BLE001
            return (f"Error saving file: {e}",)


