from .lmstudio_chat import LMStudioChat
from .lmstudio_chat_advanced import LMStudioChatAdvanced
from .tool_router import LMSToolRouter
from .http_tool_executor import LMSToolHTTPExecutor
from .loop_controller import LMStudioToolLoop
from .image_encoder import LMStudioImageEncoder
from .text_save import LMStudioSaveText


NODE_CLASS_MAPPINGS = {
    "LMStudioChat": LMStudioChat,
    "LMStudioChatAdvanced": LMStudioChatAdvanced,
    "LMSToolRouter": LMSToolRouter,
    "LMSToolHTTPExecutor": LMSToolHTTPExecutor,
    "LMStudioToolLoop": LMStudioToolLoop,
    "LMStudioImageEncoder": LMStudioImageEncoder,
    "LMStudioSaveText": LMStudioSaveText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMStudioChat": "LM Studio Chat",
    "LMStudioChatAdvanced": "LM Studio Chat (Advanced)",
    "LMSToolRouter": "LM Studio Tool Router",
    "LMSToolHTTPExecutor": "LM Studio HTTP Tool Executor",
    "LMStudioToolLoop": "LM Studio Tool Loop",
    "LMStudioImageEncoder": "LM Studio Image Encoder",
    "LMStudioSaveText": "LM Studio Save Text",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]


