import json
import os
from datetime import datetime

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args


class SaveImageTimestamp:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves images as prefix_YYYYMMDD_HHMMSS.png without 00001 counters."

    def _unique_path(self, folder, filename):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file = f"{filename}_{ts}.png"
        full = os.path.join(folder, file)
        if not os.path.exists(full):
            return file, full
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file = f"{filename}_{ts}.png"
        full = os.path.join(folder, file)
        while os.path.exists(full):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            file = f"{filename}_{ts}.png"
            full = os.path.join(folder, file)
        return file, full

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(self.output_dir, subfolder)
        os.makedirs(full_output_folder, exist_ok=True)

        results = []
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file, path = self._unique_path(full_output_folder, filename)
            img.save(path, pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })

        return {"ui": {"images": results}, "result": (images,)}


NODE_CLASS_MAPPINGS = {
    "SaveImageTimestamp": SaveImageTimestamp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageTimestamp": "Save Image (Timestamp)",
}
