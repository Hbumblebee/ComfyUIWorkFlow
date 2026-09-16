import comfy.utils


class ImageClampMaxDimension:
    upscale_methods = ["lanczos", "area", "bicubic", "bilinear", "nearest-exact"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (s.upscale_methods,),
                "max_dimension": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "clamp"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "Downscale so the longest side is at most max_dimension. Smaller images are unchanged."

    def clamp(self, image, upscale_method, max_dimension):
        height = image.shape[1]
        width = image.shape[2]
        longest = max(height, width)
        if longest <= max_dimension:
            return (image,)

        if height >= width:
            new_h = max_dimension
            new_w = max(1, round((width / height) * max_dimension))
        else:
            new_w = max_dimension
            new_h = max(1, round((height / width) * max_dimension))

        samples = image.movedim(-1, 1)
        scaled = comfy.utils.common_upscale(samples, new_w, new_h, upscale_method, "disabled")
        return (scaled.movedim(1, -1),)


NODE_CLASS_MAPPINGS = {
    "ImageClampMaxDimension": ImageClampMaxDimension,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageClampMaxDimension": "Clamp Image Max Dimension",
}
