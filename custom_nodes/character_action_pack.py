import math
import os

import torch
from PIL import Image, ImageOps
import numpy as np

import folder_paths
import node_helpers
import comfy.utils


IDENTITY_LOCK = (
    "最终画面里的人必须是人物B，不是人物A。"
    "严格保持人物B的身份、五官结构、脸型、发型、发色、肤色、瞳色、唇色、体型、年龄、"
    "服装款式与颜色、配饰与 Picture 2 / Picture 3 完全一致。"
    "肤色、头发、眼睛、嘴唇、衣服、配饰的颜色必须与人物B一致，"
    "不要自动调色、不要加滤镜、不要美白、不要改变冷暖或饱和度。"
    "保持人物B原图画风（照片保持照片，插画保持插画）。"
    "不要变成人物A，不要变成另一个人，不要添加文字、箭头或九宫格。"
)

POSE_PROMPT = (
    "人物A提供动作，人物B提供身份。Picture 1 是人物A的动作参考图；"
    "Picture 2 是人物B的全身三视图拼图（从左到右：正面、侧面、背面）；"
    "Picture 3 是人物B的脸部四特写拼图（正面、侧面、上面、下面）。\n"
    "生成人物B做出人物A的动作：姿势、肢体、构图、镜头和场景光线跟 Picture 1 走；"
    "脸、发型、体型、服装必须是人物B。不要用人物A的五官、发色或衣服。"
    "衣服随新动作自然褶皱，但款式和颜色必须仍是人物B。融合自然，光影跟 Picture 1 一致。\n"
    + IDENTITY_LOCK
)

TEXT_PROMPT = (
    "没有人物A的动作参考图，动作用下面的提示词。人物B提供身份："
    "Picture 1 是人物B正面全身，Picture 2 是人物B全身三视图（正面、侧面、背面），"
    "Picture 3 是人物B脸部四特写（正面、侧面、上面、下面）。\n"
    "生成人物B的新动作图，不要保持三视图那种直立站姿，不要把人物B换成别人。"
    "衣服随新动作自然褶皱，但款式和颜色必须仍是人物B。\n"
    "人物A的动作提示词：{action}\n"
    + IDENTITY_LOCK
)

NEG_POSE = (
    "人物A的脸, 换成动作参考图里的人, 保留人物A的五官, 人物A的发型, 人物A的衣服, "
    "标准站姿, A-pose, T-pose, 自动调色, 滤镜, 美白, 换发色, 换衣服颜色, "
    "肤色变化, color shift, recolor, filter, bleaching"
)

NEG_TEXT = (
    "标准站立转面, 标准站姿, A-pose, T-pose, 角色设定三视图, "
    "自动调色, 滤镜, 美白, 换发色, 换衣服颜色, 肤色变化, color shift, recolor, filter, bleaching"
)


def _to_1_5mp(image):
    _batch, height, width, _channels = image.shape
    pixels = height * width
    target = int(1.5 * 1_000_000)
    if pixels <= target:
        return image
    scale = math.sqrt(target / pixels)
    new_h = max(8, int(round(height * scale / 8.0) * 8))
    new_w = max(8, int(round(width * scale / 8.0) * 8))
    samples = image.movedim(-1, 1)
    scaled = comfy.utils.common_upscale(samples, new_w, new_h, "lanczos", "disabled")
    return scaled.movedim(1, -1)


def _resize_to_h(image, target_h):
    _batch, height, width, _channels = image.shape
    if height == target_h:
        return image
    new_w = max(1, int(round(width * target_h / max(height, 1))))
    samples = image.movedim(-1, 1)
    return comfy.utils.common_upscale(samples, new_w, target_h, "lanczos", "disabled").movedim(1, -1)


def _pad_w(image, target_w, fill=1.0):
    batch, height, width, channels = image.shape
    if width == target_w:
        return image
    canvas = torch.full(
        (batch, height, target_w, channels),
        fill,
        dtype=image.dtype,
        device=image.device,
    )
    x0 = max(0, (target_w - width) // 2)
    canvas[:, :, x0 : x0 + width, :] = image
    return canvas


def _hcat(images, gap=16, fill=1.0):
    target_h = max(im.shape[1] for im in images)
    aligned = [_resize_to_h(im, target_h) for im in images]
    batch, _, _, channels = aligned[0].shape
    pieces = [aligned[0]]
    for im in aligned[1:]:
        gap_t = torch.full(
            (batch, target_h, gap, channels),
            fill,
            dtype=im.dtype,
            device=im.device,
        )
        pieces.extend([gap_t, im])
    return torch.cat(pieces, dim=2)


def _vcat(top, bottom, gap=16, fill=1.0):
    target_w = max(top.shape[2], bottom.shape[2])
    top = _pad_w(top, target_w, fill)
    bottom = _pad_w(bottom, target_w, fill)
    batch, _, width, channels = top.shape
    gap_t = torch.full(
        (batch, gap, width, channels),
        fill,
        dtype=top.dtype,
        device=top.device,
    )
    return torch.cat([top, gap_t, bottom], dim=1)


class LoadOptionalImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        choices = [""] + sorted(files)
        return {
            "required": {
                "image": (choices, {"image_upload": True}),
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "has_image")
    FUNCTION = "load_image"
    DESCRIPTION = "人物A的动作参考图。留空则只用人物A的动作提示词。"

    def load_image(self, image):
        if not image:
            dummy = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
            return (dummy, False)

        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)
        img = node_helpers.pillow(ImageOps.exif_transpose, img)
        rgb = img.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,]
        return (tensor, True)

    @classmethod
    def IS_CHANGED(s, image):
        if not image:
            return ""
        image_path = folder_paths.get_annotated_filepath(image)
        m = os.path.getmtime(image_path)
        return f"{image_path}:{m}"

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not image:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


class CharacterActionPack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "front": ("IMAGE",),
                "side": ("IMAGE",),
                "back": ("IMAGE",),
                "face_front": ("IMAGE",),
                "face_side": ("IMAGE",),
                "face_above": ("IMAGE",),
                "face_below": ("IMAGE",),
                "pose": ("IMAGE",),
                "has_pose": ("BOOLEAN", {"forceInput": True}),
                "action_prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = (
        "image1",
        "image2",
        "image3",
        "canvas",
        "prompt",
        "negative",
        "denoise",
    )
    FUNCTION = "pack"
    CATEGORY = "image"
    DESCRIPTION = (
        "人物A提供动作（参考图或提示词），人物B提供多角度身份。"
        "输出人物B的新动作图。"
    )

    def pack(
        self,
        front,
        side,
        back,
        face_front,
        face_side,
        face_above,
        face_below,
        pose,
        has_pose,
        action_prompt,
    ):
        action = (action_prompt or "").strip()
        if not has_pose and not action:
            raise ValueError("没有人物A的动作参考图时，必须填写人物A的动作提示词。")

        body_sheet = _to_1_5mp(_hcat([front, side, back]))
        face_row1 = _hcat([face_front, face_side])
        face_row2 = _hcat([face_above, face_below])
        face_sheet = _to_1_5mp(_vcat(face_row1, face_row2))

        if has_pose:
            image1 = _to_1_5mp(pose)
            canvas = pose
            extra = f"\n人物A的动作补充：{action}" if action else ""
            prompt = POSE_PROMPT + extra
            negative = NEG_POSE
            denoise = 0.85
        else:
            image1 = _to_1_5mp(front)
            canvas = front
            prompt = TEXT_PROMPT.format(action=action)
            negative = NEG_TEXT
            denoise = 1.0

        return (image1, body_sheet, face_sheet, canvas, prompt, negative, denoise)


class ActionPromptText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("action_prompt",)
    FUNCTION = "emit"
    CATEGORY = "text"
    DESCRIPTION = "人物A的动作提示词。没有动作参考图时必填。"

    def emit(self, text):
        return ((text or "").strip(),)


NODE_CLASS_MAPPINGS = {
    "LoadOptionalImage": LoadOptionalImage,
    "CharacterActionPack": CharacterActionPack,
    "ActionPromptText": ActionPromptText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOptionalImage": "Load Optional Image",
    "CharacterActionPack": "Pack Character B + Action A",
    "ActionPromptText": "Action Prompt",
}
