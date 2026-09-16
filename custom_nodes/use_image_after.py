class UseImageAfter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "after": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "gate"
    CATEGORY = "image"
    DESCRIPTION = (
        "Output `image` unchanged after `after` is ready. "
        "Used to run GPU-heavy branches one after another."
    )

    def gate(self, image, after):
        return (image,)


NODE_CLASS_MAPPINGS = {
    "UseImageAfter": UseImageAfter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UseImageAfter": "Wait For Image",
}
