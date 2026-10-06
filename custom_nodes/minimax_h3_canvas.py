"""MiniMax H3 canvas helpers.

MiniMaxH3ImageToVideo stretches `first_frame` straight onto its `width` x `height`
canvas (comfy_extras/nodes_minimax_h3.py `_resize(..., crop="disabled")`). Feeding it a
fixed 16:9 size therefore squashes any input image into 16:9.

This node keeps the image's own aspect ratio and derives the model-compatible canvas
from it, using the same rule as `adapt_canvas` in comfy_extras/nodes_minimax_h3.py:

    - 768 px short edge
    - capped at 768 * 1344 = 1032192 px for the long edge
    - both sides rounded to a multiple of 32

`megapixels` then scales that native canvas down or up while keeping the ratio, the same
way the official Resolution Selector sizes the template (0.4 MP square -> 640x640).
Because the output image already matches the canvas exactly, the node's resize is an
identity operation and costs nothing.
"""

import math

import comfy.utils

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344

UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"]


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def _resize(image, width, height, upscale_method):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    return samples.movedim(1, -1)


class MiniMaxH3CanvasFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 2.0, "step": 0.01}),
                "upscale_method": (UPSCALE_METHODS,),
            },
            "optional": {
                "snap_to_multiple": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "max_pixels": ("INT", {"default": MAX_PIXELS, "min": 65536, "max": 16777216, "step": 1024}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resolve"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "MiniMax H3 canvas from the input image's aspect ratio. "
        "Sizes a 768-short-edge / 768x1344-capped canvas like the model expects, "
        "scaled to `megapixels`, snapped to `snap_to_multiple`, and returns the image "
        "already resized to it so the first frame is never stretched."
    )

    def resolve(self, image, megapixels, upscale_method, snap_to_multiple=32, max_pixels=MAX_PIXELS):
        src_h, src_w = int(image.shape[1]), int(image.shape[2])
        multiple = max(1, int(snap_to_multiple))

        canvas_w, canvas_h = adapt_canvas(src_w, src_h)

        # megapixels is a pixel budget, exactly like the Resolution Selector template
        scale = math.sqrt((megapixels * 1024 * 1024) / (canvas_w * canvas_h))
        canvas_w = max(multiple, round(canvas_w * scale / multiple) * multiple)
        canvas_h = max(multiple, round(canvas_h * scale / multiple) * multiple)

        cap = max(1, int(max_pixels))
        if canvas_w * canvas_h > cap:
            shrink = math.sqrt(cap / (canvas_w * canvas_h))
            canvas_w = max(multiple, round(canvas_w * shrink / multiple) * multiple)
            canvas_h = max(multiple, round(canvas_h * shrink / multiple) * multiple)

        if canvas_w != src_w or canvas_h != src_h:
            image = _resize(image, canvas_w, canvas_h, upscale_method)

        return {
            "ui": {"text": [f"{src_w}x{src_h}  ->  {canvas_w}x{canvas_h}  ({megapixels} MP)"]},
            "result": (image, canvas_w, canvas_h),
        }


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3CanvasFromImage": MiniMaxH3CanvasFromImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CanvasFromImage": "MiniMax H3 Canvas From Image",
}
