import torch
import comfy.utils


class ImagePadToCanvas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1536, "min": 64, "max": 4096, "step": 16}),
                "mode": (["auto_fullbody", "center"],),
                "fill": ("INT", {"default": 217, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pad"
    CATEGORY = "image"
    DESCRIPTION = (
        "Fit the input onto a fixed canvas. auto_fullbody puts short/bust images "
        "in the top half so the lower canvas can be outpainted into legs."
    )

    def pad(self, image, width, height, mode, fill):
        _batch, src_h, src_w, _channels = image.shape
        fill_v = fill / 255.0
        canvas = torch.full(
            (image.shape[0], height, width, image.shape[3]),
            fill_v,
            dtype=image.dtype,
            device=image.device,
        )

        src_aspect = src_h / max(src_w, 1)
        if mode == "center" or src_aspect >= 1.45:
            box_w, box_h = width, height
            top_margin = 0
            center_y = True
        else:
            box_w, box_h = width, max(64, int(height * 0.50))
            top_margin = max(0, int(height * 0.04))
            center_y = False

        scale = min(box_w / max(src_w, 1), box_h / max(src_h, 1))
        new_w = max(1, min(box_w, int(round(src_w * scale))))
        new_h = max(1, min(box_h, int(round(src_h * scale))))

        samples = image.movedim(-1, 1)
        scaled = comfy.utils.common_upscale(samples, new_w, new_h, "lanczos", "disabled")
        scaled = scaled.movedim(1, -1)

        x0 = max(0, (width - new_w) // 2)
        if center_y:
            y0 = max(0, (height - new_h) // 2)
        else:
            y0 = min(top_margin, height - new_h)
        x1 = min(width, x0 + new_w)
        y1 = min(height, y0 + new_h)
        canvas[:, y0:y1, x0:x1, :] = scaled[:, : y1 - y0, : x1 - x0, :]
        return (canvas,)


class ImageFlipX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "flip"
    CATEGORY = "image"
    DESCRIPTION = "Horizontal flip. Used to get the true opposite side of a character."

    def flip(self, image):
        return (torch.flip(image, dims=[2]),)


NODE_CLASS_MAPPINGS = {
    "ImagePadToCanvas": ImagePadToCanvas,
    "ImageFlipX": ImageFlipX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePadToCanvas": "Pad Image To Canvas",
    "ImageFlipX": "Flip Image X",
}
