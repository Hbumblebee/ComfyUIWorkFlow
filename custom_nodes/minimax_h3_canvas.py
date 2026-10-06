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
                "long_edge": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 32,
                    "tooltip": "Hard cap on the long side (0 = off). Set 1344 to lock the model's native ceiling.",
                }),
                "native_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clamp to H3's trained canvas (768 short edge, 768x1344 cap) instead of scaling a bigger canvas from megapixels.",
                }),
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

    def resolve(self, image, megapixels, upscale_method, snap_to_multiple=32, max_pixels=MAX_PIXELS,
                long_edge=0, native_only=False):
        src_h, src_w = int(image.shape[1]), int(image.shape[2])
        multiple = max(1, int(snap_to_multiple))

        canvas_w, canvas_h = adapt_canvas(src_w, src_h)
        native_w, native_h = canvas_w, canvas_h

        # megapixels is a pixel budget, exactly like the Resolution Selector template
        scale = math.sqrt((megapixels * 1024 * 1024) / (canvas_w * canvas_h))
        if native_only:
            scale = min(1.0, scale)
        canvas_w = max(multiple, round(canvas_w * scale / multiple) * multiple)
        canvas_h = max(multiple, round(canvas_h * scale / multiple) * multiple)

        cap = max(1, int(max_pixels))
        if canvas_w * canvas_h > cap:
            shrink = math.sqrt(cap / (canvas_w * canvas_h))
            canvas_w = max(multiple, round(canvas_w * shrink / multiple) * multiple)
            canvas_h = max(multiple, round(canvas_h * shrink / multiple) * multiple)

        # keep the ratio while pulling the long side under the requested ceiling
        cap_edge = int(long_edge)
        longest = max(canvas_w, canvas_h)
        if cap_edge > 0 and longest > cap_edge and longest > multiple:
            shrink = cap_edge / longest
            canvas_w = max(multiple, int(canvas_w * shrink / multiple) * multiple)
            canvas_h = max(multiple, int(canvas_h * shrink / multiple) * multiple)

        if canvas_w != src_w or canvas_h != src_h:
            image = _resize(image, canvas_w, canvas_h, upscale_method)

        label = f"{src_w}x{src_h}  ->  {canvas_w}x{canvas_h}  ({canvas_w * canvas_h / 1024 / 1024:.3f} MP)"
        if canvas_w > native_w or canvas_h > native_h:
            label += "  |  over H3 native canvas - enable native_only or lower megapixels"
        return {"ui": {"text": [label]}, "result": (image, canvas_w, canvas_h)}


FIT_MODES = ["unify_crop", "cover", "stretch"]


def _crop_to_aspect(image, target_aspect, align):
    """Center/edge crop so the frame's aspect matches the shared canvas."""
    src_h, src_w = int(image.shape[1]), int(image.shape[2])
    src_aspect = src_w / src_h
    if abs(src_aspect - target_aspect) < 1e-6:
        return image

    if src_aspect > target_aspect:
        new_w, new_h = max(1, round(src_h * target_aspect)), src_h
    else:
        new_w, new_h = src_w, max(1, round(src_w / target_aspect))

    x = (src_w - new_w) // 2
    if align == "top":
        y = 0
    elif align == "bottom":
        y = src_h - new_h
    else:
        y = (src_h - new_h) // 2
    return image[:, max(0, y):max(0, y) + new_h, max(0, x):max(0, x) + new_w, :]


def _fit_frame(image, width, height, crop):
    """Map a frame onto the canvas: `crop` picks center/top/bottom framing
    (aspect fills the canvas, overflow is cut) and "disabled" stretches.
    The frame always comes back exactly `width` x `height`."""
    samples = image[..., :3].movedim(-1, 1)
    if samples.shape[2] != height or samples.shape[3] != width:
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


class MiniMaxH3FirstLastCanvas:
    """Shared canvas for first_frame + last_frame.

    MiniMaxH3ImageToVideo handles the two keyframes differently: the first frame
    is *stretched* onto the canvas while the last frame is *center-cropped*. Feed
    it two raw images and the pair only lines up when they agree with the canvas
    shape. This node picks one canvas and brings both frames to it identically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_frame": ("IMAGE",),
                "fit": (FIT_MODES,),
                "megapixels": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 2.0, "step": 0.01}),
                "crop_align": (["center", "top", "bottom"],),
            },
            "optional": {
                "last_frame": ("IMAGE",),
                "snap_to_multiple": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "max_pixels": ("INT", {"default": MAX_PIXELS, "min": 65536, "max": 16777216, "step": 1024}),
                "long_edge": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "native_only": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("first_image", "last_image", "width", "height")
    FUNCTION = "resolve"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "First/last-frame canvas for MiniMax H3 fl2va. Sizes one canvas from first_frame, "
        "brings both keyframes onto it the same way, and returns width/height for the "
        "MiniMaxH3ImageToVideo node. fit: unify_crop keeps both poses exactly (default), "
        "cover fills the frame (may crop), stretch fills without cropping (distorts)."
    )

    def resolve(self, first_frame, fit, megapixels, crop_align, last_frame=None,
                snap_to_multiple=32, max_pixels=MAX_PIXELS, long_edge=0, native_only=True):
        multiple = max(1, int(snap_to_multiple))
        f_h, f_w = int(first_frame.shape[1]), int(first_frame.shape[2])

        # canvas geometry comes from the first frame; the last frame only follows
        canvas_w, canvas_h = adapt_canvas(f_w, f_h)
        native_w, native_h = canvas_w, canvas_h

        scale = math.sqrt((megapixels * 1024 * 1024) / (canvas_w * canvas_h))
        if native_only:
            scale = min(1.0, scale)
        canvas_w = max(multiple, round(canvas_w * scale / multiple) * multiple)
        canvas_h = max(multiple, round(canvas_h * scale / multiple) * multiple)

        cap = max(1, int(max_pixels))
        if canvas_w * canvas_h > cap:
            shrink = math.sqrt(cap / (canvas_w * canvas_h))
            canvas_w = max(multiple, round(canvas_w * shrink / multiple) * multiple)
            canvas_h = max(multiple, round(canvas_h * shrink / multiple) * multiple)

        cap_edge = int(long_edge)
        longest = max(canvas_w, canvas_h)
        if cap_edge > 0 and longest > cap_edge and longest > multiple:
            shrink = cap_edge / longest
            canvas_w = max(multiple, int(canvas_w * shrink / multiple) * multiple)
            canvas_h = max(multiple, int(canvas_h * shrink / multiple) * multiple)

        target_aspect = canvas_w / canvas_h
        last = None
        note = ""
        if last_frame is not None:
            l_h, l_w = int(last_frame.shape[1]), int(last_frame.shape[2])
            if abs(l_w / l_h - f_w / f_h) > 0.05:
                note = f"  |  ! first {f_w}x{f_h} vs last {l_w}x{l_h}: shapes differ"
            if fit == "unify_crop":
                # crop both frames just enough to cover the shared canvas shape; the
                # node's own center-crop is then a no-op, so neither pose moves
                first = _fit_frame(_crop_to_aspect(first_frame[:1], target_aspect, crop_align),
                                   canvas_w, canvas_h, "disabled")
                last = _fit_frame(_crop_to_aspect(last_frame[:1], target_aspect, crop_align),
                                  canvas_w, canvas_h, "disabled")
            elif fit == "cover":
                # stock behaviour: aspect fills the canvas, overflow is cut
                first = _fit_frame(first_frame[:1], canvas_w, canvas_h, "disabled")
                last = _fit_frame(last_frame[:1], canvas_w, canvas_h, crop_align)
            else:
                # stretch: nothing is lost, the aspect ratio is
                first = _fit_frame(first_frame[:1], canvas_w, canvas_h, "disabled")
                last = _fit_frame(last_frame[:1], canvas_w, canvas_h, "disabled")
        else:
            first = _fit_frame(first_frame[:1], canvas_w, canvas_h, "disabled")
            note = "  |  no last_frame connected"

        label = (f"first {f_w}x{f_h} -> {canvas_w}x{canvas_h} "
                 f"({canvas_w * canvas_h / 1024 / 1024:.3f} MP)  [{fit}]{note}")
        if canvas_w > native_w or canvas_h > native_h:
            label += "  |  over H3 native canvas"
        return {"ui": {"text": [label]}, "result": (first, last, canvas_w, canvas_h)}


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3CanvasFromImage": MiniMaxH3CanvasFromImage,
    "MiniMaxH3FirstLastCanvas": MiniMaxH3FirstLastCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CanvasFromImage": "MiniMax H3 Canvas From Image",
    "MiniMaxH3FirstLastCanvas": "MiniMax H3 First/Last Canvas",
}
