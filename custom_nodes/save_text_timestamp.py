import os
from datetime import datetime

import folder_paths


class SaveTextTimestamp:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "PromptReverse"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "text"
    DESCRIPTION = "Saves text as prefix_YYYYMMDD_HHMMSS.txt without 00001 counters."

    def _unique_path(self, folder, filename):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file = f"{filename}_{ts}.txt"
        full = os.path.join(folder, file)
        if not os.path.exists(full):
            return file, full
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file = f"{filename}_{ts}.txt"
        full = os.path.join(folder, file)
        while os.path.exists(full):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            file = f"{filename}_{ts}.txt"
            full = os.path.join(folder, file)
        return file, full

    def save_text(self, text, filename_prefix="PromptReverse"):
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(self.output_dir, subfolder)
        os.makedirs(full_output_folder, exist_ok=True)

        file, path = self._unique_path(full_output_folder, filename)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text if text is not None else "")

        return {
            "ui": {"text": (text,)},
            "result": (text,),
        }


NODE_CLASS_MAPPINGS = {
    "SaveTextTimestamp": SaveTextTimestamp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveTextTimestamp": "Save Text (Timestamp)",
}
