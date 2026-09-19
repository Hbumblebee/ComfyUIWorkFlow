import torch


def _as_mask(mask):
    if mask.ndim == 4:
        return mask[:, :, :, 0]
    return mask


def _has_transparency(mask):
    if mask is None:
        return False
    return float(_as_mask(mask).max().item()) > 0.002


def _resize_mask(mask, height, width):
    mask = _as_mask(mask)
    if mask.shape[-2] == height and mask.shape[-1] == width:
        return mask
    resized = torch.nn.functional.interpolate(
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized[:, 0]


def _fill_tensor(image, fill):
    color = torch.zeros_like(image[..., :3])
    color[..., 0] = ((fill >> 16) & 0xFF) / 255.0
    color[..., 1] = ((fill >> 8) & 0xFF) / 255.0
    color[..., 2] = (fill & 0xFF) / 255.0
    return color


class CompositeCutoutOnFill:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill": (
                    "INT",
                    {
                        "default": 0xD9D9D9,
                        "min": 0,
                        "max": 0xFFFFFF,
                        "step": 1,
                        "display": "color",
                    },
                ),
                "mask_mode": (["LoadImage透明区", "前景遮罩"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "composite"
    CATEGORY = "image/compositing"
    DESCRIPTION = (
        "LoadImage drops PNG alpha and keeps leftover RGB, so Qwen Edit sees the old "
        "background. This node paints transparent pixels with a solid fill and returns "
        "a LoadImage-style mask (1 = transparent)."
    )

    def composite(self, image, mask, fill, mask_mode):
        rgb = image[..., :3]
        height, width = rgb.shape[1], rgb.shape[2]
        if mask_mode == "LoadImage透明区" and not _has_transparency(mask):
            return (rgb, mask)

        keep = _resize_mask(mask.to(device=rgb.device, dtype=rgb.dtype), height, width)
        if mask_mode == "LoadImage透明区":
            keep = 1.0 - keep
        keep = keep.unsqueeze(-1).clamp(0.0, 1.0)
        out = rgb * keep + _fill_tensor(rgb, int(fill)) * (1.0 - keep)
        load_mask = (1.0 - keep.squeeze(-1)).clamp(0.0, 1.0)
        return (out, load_mask)


class ApplyLoadImageAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "restore_alpha": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/compositing"
    DESCRIPTION = (
        "Put LoadImage alpha back after an RGB model (Qwen Edit / VAE) so the saved PNG "
        "stays transparent. Transparent RGB is zeroed so the old scene cannot leak."
    )

    def apply(self, image, mask, restore_alpha=True):
        rgb = image[..., :3]
        if not restore_alpha or not _has_transparency(mask):
            return (rgb,)

        height, width = rgb.shape[1], rgb.shape[2]
        load_mask = _resize_mask(mask.to(device=rgb.device, dtype=rgb.dtype), height, width)
        alpha = (1.0 - load_mask).clamp(0.0, 1.0).unsqueeze(-1)
        rgb = rgb * alpha
        return (torch.cat([rgb, alpha], dim=-1),)


NODE_CLASS_MAPPINGS = {
    "CompositeCutoutOnFill": CompositeCutoutOnFill,
    "ApplyLoadImageAlpha": ApplyLoadImageAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompositeCutoutOnFill": "Composite Cutout On Fill 抠图铺底色",
    "ApplyLoadImageAlpha": "Apply LoadImage Alpha 贴回透明通道",
}
