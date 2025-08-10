import base64
import io
from typing import Any, Dict, List, Tuple

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


class LMStudioImageEncoder:
    CATEGORY = "LM Studio"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("data_url",)
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "format": (["PNG", "JPEG", "WEBP"], {"default": "PNG"}),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
            },
            "optional": {
                "strip_alpha": ("BOOLEAN", {"default": False}),
            },
        }

    def encode(
        self,
        image: Any,
        format: str = "PNG",
        quality: int = 90,
        strip_alpha: bool = False,
    ) -> Tuple[str]:
        if Image is None or np is None:
            return ("",)
        try:
            if hasattr(image, "cpu"):
                arr = image.cpu().numpy()
            else:
                arr = image
            # Expect [B,H,W,C] or [H,W,C]
            img_np = arr[0] if len(arr.shape) == 4 else arr
            img_np = (img_np * 255.0).clip(0, 255).astype("uint8")
            if img_np.shape[-1] == 1:
                img_np = img_np[..., 0]
            pil = Image.fromarray(img_np)
            if strip_alpha and pil.mode in ("RGBA", "LA"):
                pil = pil.convert("RGB")
            buf = io.BytesIO()
            save_kwargs = {}
            if format == "JPEG":
                pil = pil.convert("RGB")
                save_kwargs["quality"] = int(quality)
                save_kwargs["optimize"] = True
            elif format == "WEBP":
                save_kwargs["quality"] = int(quality)
            pil.save(buf, format=format, **save_kwargs)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            mime = {
                "PNG": "image/png",
                "JPEG": "image/jpeg",
                "WEBP": "image/webp",
            }.get(format, "image/png")
            return (f"data:{mime};base64,{b64}",)
        except Exception:
            return ("",)


