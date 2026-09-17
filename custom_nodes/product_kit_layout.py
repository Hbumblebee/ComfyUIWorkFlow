import hashlib
import math
import os

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

NONE_CHOICE = "（不选）"
NONE_VALUES = {"", NONE_CHOICE, "(none)", "none"}


SLOT_COUNT = 25
STACK_MODES = ["横排层叠", "圆片层叠", "并排", "双层叠放", "两列堆叠"]
BG_STYLES = ["紫蓝渐变", "蓝白渐变", "纯白", "自定义底图"]
CUTOUT_MODES = ["遮罩前景", "LoadImage遮罩", "自动去底"]

_LABEL_FILL = (255, 22, 22, 255)
_LABEL_STROKE = (255, 255, 255, 255)
_FOOTER_C0 = np.array([90, 68, 196], dtype=np.float32)
_FOOTER_C1 = np.array([58, 122, 232], dtype=np.float32)


def _is_none_choice(name):
    return name is None or str(name).strip() in NONE_VALUES


def is_placeholder_image(image):
    if image is None:
        return True
    if not torch.is_tensor(image):
        return True
    if image.ndim < 3:
        return True
    return int(image.shape[-3]) <= 1 and int(image.shape[-2]) <= 1


def empty_image_mask():
    image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    mask = torch.ones((1, 1, 1), dtype=torch.float32)
    return (image, mask)


def coerce_qty(qty):
    if qty is None or qty is False:
        return 0
    if isinstance(qty, bool):
        return 0
    if isinstance(qty, (int, float)):
        n = int(qty)
        return n if n > 0 else 0
    text = str(qty).strip()
    if text in NONE_VALUES or text in ("image", "mask", "video"):
        return 0
    try:
        n = int(float(text))
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _first_hwc(t):
    arr = t.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def _to_uint8(arr):
    if arr.dtype in (np.float32, np.float16, np.float64):
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _resize_mask(mask_hw, size):
    w, h = size
    im = Image.fromarray(_to_uint8(mask_hw), mode="L")
    if im.size != (w, h):
        im = im.resize((w, h), Image.Resampling.BILINEAR)
    return np.array(im)


def _auto_key(rgb):
    h, w = rgb.shape[:2]
    ch = max(1, min(12, h // 16))
    cw = max(1, min(12, w // 16))
    corners = np.concatenate(
        [
            rgb[:ch, :cw].reshape(-1, 3),
            rgb[:ch, -cw:].reshape(-1, 3),
            rgb[-ch:, :cw].reshape(-1, 3),
            rgb[-ch:, -cw:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners.astype(np.float32), axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=-1)
    lum = rgb.astype(np.float32).max(axis=-1)
    sat = rgb.astype(np.float32).max(axis=-1) - rgb.astype(np.float32).min(axis=-1)
    near_bg = np.clip((dist - 14.0) / 26.0, 0.0, 1.0)
    not_white = np.clip((248.0 - lum) / 18.0, 0.0, 1.0)
    has_chroma = np.clip(sat / 18.0, 0.0, 1.0)
    alpha = np.maximum(near_bg, not_white * has_chroma)
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def tensor_to_rgba(image, mask, cutout):
    arr = _first_hwc(image)
    rgb = _to_uint8(arr[..., :3])
    h, w = rgb.shape[:2]
    alpha = None
    if arr.shape[-1] >= 4:
        alpha = _to_uint8(arr[..., 3])

    if mask is not None:
        m = _first_hwc(mask)
        if m.ndim == 3:
            m = m[..., 0]
        m8 = _resize_mask(m, (w, h))
        if cutout == "LoadImage遮罩":
            alpha = 255 - m8
        else:
            alpha = m8
    elif alpha is None:
        alpha = _auto_key(rgb)
    elif cutout == "自动去底":
        alpha = np.minimum(alpha, _auto_key(rgb))

    if alpha is None or int(alpha.max()) < 8:
        alpha = _auto_key(rgb)

    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def trim_rgba(im, threshold=10):
    a = np.array(im.getchannel("A"))
    ys, xs = np.where(a > threshold)
    if xs.size == 0 or ys.size == 0:
        return im
    pad = 2
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(im.width, int(xs.max()) + 1 + pad)
    y1 = min(im.height, int(ys.max()) + 1 + pad)
    return im.crop((x0, y0, x1, y1))


def feather_alpha(im, radius=1.2):
    if radius <= 0:
        return im
    r, g, b, a = im.split()
    a = a.filter(ImageFilter.GaussianBlur(radius))
    out = Image.merge("RGBA", (r, g, b, a))
    return out


def _find_font(size, bold=False, black=False):
    if black:
        names = [
            r"C:\Windows\Fonts\ariblk.ttf",
            r"C:\Windows\Fonts\impact.ttf",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\arialbd.ttf",
        ]
    elif bold:
        names = [
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
    else:
        names = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\simhei.ttf",
        ]
    for path in names:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def _scale_rgba(im, sw, sh):
    sw = max(1, int(round(sw)))
    sh = max(1, int(round(sh)))
    if im.size == (sw, sh):
        return im
    return im.resize((sw, sh), Image.Resampling.LANCZOS)


def alpha_paste(base, overlay, xy):
    x, y = int(round(xy[0])), int(round(xy[1]))
    bw, bh = base.size
    ow, oh = overlay.size
    if ow <= 0 or oh <= 0:
        return
    x0, y0 = x, y
    x1, y1 = x + ow, y + oh
    if x1 <= 0 or y1 <= 0 or x0 >= bw or y0 >= bh:
        return
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(bw, x1), min(bh, y1)
    crop = overlay.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
    base.alpha_composite(crop, dest=(cx0, cy0))


def paste_with_shadow(base, sprite, xy, enabled):
    x, y = xy
    if enabled:
        blur = max(4, int(round(min(sprite.size) * 0.035)))
        ox = max(2, blur // 3)
        oy = max(3, blur // 2)
        pad = blur * 2
        layer = Image.new("RGBA", (sprite.width + pad * 2, sprite.height + pad * 2), (0, 0, 0, 0))
        shadow = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
        shade = sprite.split()[-1].point(lambda p: int(p * 0.38))
        shadow.putalpha(shade)
        layer.paste(shadow, (pad, pad), shadow)
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
        alpha_paste(base, layer, (x - pad + ox, y - pad + oy))
    alpha_paste(base, sprite, (x, y))


_CELL_FILL = 0.97
_MIN_CELL = 36
_STACK_GAP_RATIO = 0.08


def _mode_spec(mode):
    if mode == "并排":
        return "pair", 0.0
    if mode == "圆片层叠":
        return "row", 0.62
    if mode == "双层叠放":
        return "two_row", 0.5
    if mode == "两列堆叠":
        return "two_col", 0.7
    return "row", 0.5


def _fan_row_native(sw, sh, n, overlap):
    n = max(1, int(n))
    span = 1.0 if n == 1 else 1.0 + (n - 1) * (1.0 - overlap)
    return sw * span, sh


def _fan_col_native(sw, sh, n, overlap):
    n = max(1, int(n))
    span = 1.0 if n == 1 else 1.0 + (n - 1) * (1.0 - overlap)
    return sw, sh * span


def _stack_native_size(sprite, copies, mode):
    sw = float(max(getattr(sprite, "width", 1), 1))
    sh = float(max(getattr(sprite, "height", 1), 1))
    n = max(1, int(copies))
    kind, overlap = _mode_spec(mode)
    if kind == "pair":
        show = 1 if n <= 1 else 2
        gap = _STACK_GAP_RATIO * sw
        return show * sw + (show - 1) * gap, sh
    if kind == "two_row":
        top_n = (n + 1) // 2
        bot_n = n - top_n
        bw, bh = _fan_row_native(sw, sh, top_n, overlap)
        if bot_n <= 0:
            return bw, bh
        bw2, bh2 = _fan_row_native(sw, sh, bot_n, overlap)
        return max(bw, bw2), bh + bh2 + _STACK_GAP_RATIO * sh
    if kind == "two_col":
        left_n = (n + 1) // 2
        right_n = n - left_n
        bw, bh = _fan_col_native(sw, sh, left_n, overlap)
        if right_n <= 0:
            return bw, bh
        bw2, bh2 = _fan_col_native(sw, sh, right_n, overlap)
        return bw + bw2 + _STACK_GAP_RATIO * sw, max(bh, bh2)
    return _fan_row_native(sw, sh, n, overlap)


def _fan_metrics(sprite, n, region, overlap):
    _x, _y, rw, rh = region
    n = max(1, n)
    max_w = max(1.0, rw * _CELL_FILL)
    max_h = max(1.0, rh * _CELL_FILL)
    ratio_h_w = sprite.height / max(sprite.width, 1)
    span = 1.0 if n == 1 else 1.0 + (n - 1) * (1.0 - overlap)
    sw = min(max_w / span, max_h / max(ratio_h_w, 1e-6))
    sh = sw * ratio_h_w
    step = 0.0 if n == 1 else sw * (1.0 - overlap)
    total = sw if n == 1 else sw + step * (n - 1)
    return sw, sh, step, total


def draw_fan_row(canvas, sprite, n, region, overlap, shadow):
    x, y, rw, rh = region
    sw, sh, step, total = _fan_metrics(sprite, n, region, overlap)
    scaled = _scale_rgba(sprite, sw, sh)
    ox = x + (rw - total) / 2.0
    oy = y + (rh - sh) / 2.0
    for i in range(n):
        paste_with_shadow(canvas, scaled, (ox + i * step, oy), shadow)


def draw_fan_col(canvas, sprite, n, region, overlap, shadow):
    x, y, rw, rh = region
    n = max(1, n)
    max_w = max(1.0, rw * _CELL_FILL)
    max_h = max(1.0, rh * _CELL_FILL)
    ratio_w_h = sprite.width / max(sprite.height, 1)
    span = 1.0 if n == 1 else 1.0 + (n - 1) * (1.0 - overlap)
    sh = min(max_h / span, max_w / max(ratio_w_h, 1e-6))
    sw = sh * ratio_w_h
    step = 0.0 if n == 1 else sh * (1.0 - overlap)
    total = sh if n == 1 else sh + step * (n - 1)
    scaled = _scale_rgba(sprite, sw, sh)
    ox = x + (rw - sw) / 2.0
    oy = y + (rh - total) / 2.0
    for i in range(n):
        paste_with_shadow(canvas, scaled, (ox, oy + i * step), shadow)


def draw_pair(canvas, sprite, n, region, shadow):
    x, y, rw, rh = region
    show = 1 if n <= 1 else 2
    gap = max(4.0, rw * 0.03)
    max_w = max(1.0, rw * _CELL_FILL)
    max_h = max(1.0, rh * _CELL_FILL)
    ratio = sprite.width / max(sprite.height, 1)
    sh = max_h
    sw = sh * ratio
    total = show * sw + (show - 1) * gap
    if total > max_w:
        sw = max(1.0, (max_w - (show - 1) * gap) / show)
        sh = sw / max(ratio, 1e-6)
        total = show * sw + (show - 1) * gap
    scaled = _scale_rgba(sprite, sw, sh)
    ox = x + (rw - total) / 2.0
    oy = y + (rh - sh) / 2.0
    for i in range(show):
        paste_with_shadow(canvas, scaled, (ox + i * (sw + gap), oy), shadow)


def draw_stack(canvas, sprite, count, mode, region, shadow):
    kind, overlap = _mode_spec(mode)
    if kind == "pair":
        draw_pair(canvas, sprite, count, region, shadow)
        return
    if kind == "two_row":
        x, y, w, h = region
        top_n = (count + 1) // 2
        bot_n = count - top_n
        if bot_n <= 0:
            draw_fan_row(canvas, sprite, top_n, region, overlap=overlap, shadow=shadow)
            return
        gap = max(6, int(h * 0.04))
        h1 = (h - gap) // 2
        draw_fan_row(canvas, sprite, top_n, (x, y, w, h1), overlap=overlap, shadow=shadow)
        draw_fan_row(
            canvas,
            sprite,
            bot_n,
            (x, y + h1 + gap, w, h - h1 - gap),
            overlap=overlap,
            shadow=shadow,
        )
        return
    if kind == "two_col":
        x, y, w, h = region
        left_n = (count + 1) // 2
        right_n = count - left_n
        if right_n <= 0:
            draw_fan_col(canvas, sprite, left_n, region, overlap=overlap, shadow=shadow)
            return
        gap = max(6, int(w * 0.04))
        w1 = (w - gap) // 2
        draw_fan_col(canvas, sprite, left_n, (x, y, w1, h), overlap=overlap, shadow=shadow)
        draw_fan_col(
            canvas,
            sprite,
            right_n,
            (x + w1 + gap, y, w - w1 - gap, h),
            overlap=overlap,
            shadow=shadow,
        )
        return
    draw_fan_row(canvas, sprite, count, region, overlap=overlap, shadow=shadow)


def _split_weighted(origin, length, weights, gap):
    n = len(weights)
    if n <= 0:
        return []
    gap = int(gap)
    origin = int(origin)
    length = int(length)
    usable = length - gap * (n - 1)
    if usable < n:
        usable = n
    total = float(sum(max(w, 1e-6) for w in weights))
    raw = [usable * max(w, 1e-6) / total for w in weights]
    sizes = [max(1, int(round(x))) for x in raw]
    diff = usable - sum(sizes)
    order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=diff > 0)
    idx = 0
    guard = 0
    while diff != 0 and guard < usable + n + 8:
        i = order[idx % n]
        if diff > 0:
            sizes[i] += 1
            diff -= 1
        elif sizes[i] > 1:
            sizes[i] -= 1
            diff += 1
        idx += 1
        guard += 1
    pos = origin
    spans = []
    for size in sizes:
        spans.append((pos, size))
        pos += size + gap
    return spans


def _inner_frame(width, height, footer_h):
    mx = max(10, int(width * 0.018))
    my = max(8, int(height * 0.016))
    gap = max(8, int(min(width, height) * 0.012))
    inner_w = width - 2 * mx
    inner_h = height - footer_h - my - max(my, 8)
    return mx, my, gap, inner_w, inner_h


def _row_natural_height(aspects, inner_w, gap):
    m = len(aspects)
    sum_a = sum(max(a, 0.05) for a in aspects)
    return (inner_w - gap * (m - 1)) / max(sum_a, 1e-6)


def _wrap_rows(aspects, inner_w, gap, target_h):
    min_w = max(_MIN_CELL, inner_w * 0.13)

    def group_ok(idxs):
        group = [max(aspects[i], 0.05) for i in idxs]
        sum_a = sum(group)
        usable = inner_w - gap * (len(group) - 1)
        return min(usable * a / sum_a for a in group) >= min_w - 0.5

    rows = []
    row = []
    width = 0.0
    for i, aspect in enumerate(aspects):
        item_w = max(aspect, 0.05) * target_h
        extra = 0.0 if not row else float(gap)
        would = row + [i]
        too_wide = row and width + extra + item_w > inner_w + 0.5
        too_thin = row and not group_ok(would)
        if too_wide or too_thin:
            rows.append(len(row))
            row = [i]
            width = item_w
        else:
            width += extra + item_w
            row.append(i)
    if row:
        rows.append(len(row))
    return rows


def _layout_quality(counts, aspects, natives, inner_w, inner_h, gap):
    n = len(aspects)
    if not counts or sum(counts) != n:
        return None
    idx = 0
    row_hs = []
    groups = []
    for cols in counts:
        group = aspects[idx : idx + cols]
        row_hs.append(max(_row_natural_height(group, inner_w, gap), 1.0))
        groups.append((idx, cols, group))
        idx += cols
    usable = inner_h - gap * (len(counts) - 1)
    if usable < _MIN_CELL:
        return None
    k = usable / max(sum(row_hs), 1e-6)
    log_area = 0.0
    min_area = float("inf")
    min_side = float("inf")
    idx = 0
    for rh, (_start, cols, group) in zip(row_hs, groups):
        cell_h = rh * k
        sum_a = sum(max(a, 0.05) for a in group)
        usable_w = inner_w - gap * (cols - 1)
        for j, aspect in enumerate(group):
            cell_w = usable_w * max(aspect, 0.05) / sum_a
            bw, bh = natives[idx]
            scale = min(cell_w / max(bw, 1e-6), cell_h / max(bh, 1e-6))
            dw = bw * scale
            dh = bh * scale
            if dw > cell_w + 1.0 or dh > cell_h + 1.0:
                return None
            area = max(dw * dh, 1.0)
            log_area += math.log(area)
            min_area = min(min_area, area)
            min_side = min(min_side, dw, dh)
            idx += 1
    if min_side < _MIN_CELL:
        min_area -= (_MIN_CELL - min_side) ** 2
    return (min_area, log_area, min_side, -abs(1.0 - min(k, 1.0)))


def _pack_row_breaks(aspects, natives, inner_w, inner_h, gap):
    n = len(aspects)
    if n <= 1:
        return [n]
    heights = [inner_h / r for r in range(1, n + 1)]
    widest = max(aspects) if aspects else 1.0
    heights.append(inner_w / max(widest, 0.05))
    for step in range(6, 72):
        heights.append(max(24.0, inner_h * step / 72.0))
    for _bw, bh in natives:
        heights.append(max(24.0, bh))
        heights.append(max(24.0, bh * 1.4))
    best_counts = [n]
    best_score = None
    seen = set()
    for target_h in heights:
        counts = tuple(_wrap_rows(aspects, inner_w, gap, max(24.0, target_h)))
        if counts in seen:
            continue
        seen.add(counts)
        score = _layout_quality(list(counts), aspects, natives, inner_w, inner_h, gap)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_counts = list(counts)
    if best_score is None:
        return [n]
    return best_counts


def _pack_regions(slots, width, height, footer_h):
    n = len(slots)
    mx, my, gap, inner_w, inner_h = _inner_frame(width, height, footer_h)
    if n <= 0:
        return []
    natives = [
        _stack_native_size(slot["sprite"], slot["copies"], slot["mode"])
        for slot in slots
    ]
    aspects = [max(bw / max(bh, 1e-6), 0.05) for bw, bh in natives]
    counts = _pack_row_breaks(aspects, natives, inner_w, inner_h, gap)
    row_heights = []
    row_weights = []
    idx = 0
    for cols in counts:
        group = aspects[idx : idx + cols]
        row_heights.append(max(_row_natural_height(group, inner_w, gap), 1.0))
        row_weights.append(group)
        idx += cols
    y_spans = _split_weighted(my, inner_h, row_heights, gap)
    boxes = []
    for weights, (y, cell_h) in zip(row_weights, y_spans):
        x_spans = _split_weighted(mx, inner_w, weights, gap)
        for x, cell_w in x_spans:
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width, x + max(_MIN_CELL, cell_w))
            y1 = min(max(y0 + 1, height - footer_h), y + max(_MIN_CELL, cell_h))
            boxes.append((x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
    return boxes


def _regions(n, width, height, footer_h):
    dummy = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
    slots = [
        {"sprite": dummy, "copies": 1, "mode": "横排层叠"}
        for _ in range(max(1, int(n)))
    ]
    return _pack_regions(slots, width, height, footer_h)


def _gradient_v(width, height, colors):
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)
    rgb = np.zeros((height, 3), dtype=np.float32)
    stops = len(colors) - 1
    for i, u in enumerate(t):
        x = u * stops
        k = min(stops - 1, int(x))
        f = x - k
        rgb[i] = colors[k] * (1.0 - f) + colors[k + 1] * f
    plane = np.broadcast_to(rgb[:, None, :], (height, width, 3))
    return Image.fromarray(plane.astype(np.uint8), "RGB").convert("RGBA")


def _gradient_h(width, height, c0, c1):
    t = np.linspace(0.0, 1.0, width, dtype=np.float32)
    rgb = c0[None, :] * (1.0 - t[:, None]) + c1[None, :] * t[:, None]
    plane = np.broadcast_to(rgb[None, :, :], (height, width, 3))
    return Image.fromarray(plane.astype(np.uint8), "RGB").convert("RGBA")


def make_background(width, height, style, custom):
    if custom is not None and not is_placeholder_image(custom):
        arr = _first_hwc(custom)
        rgb = _to_uint8(arr[..., :3])
        img = Image.fromarray(rgb, "RGB")
        scale = max(width / max(img.width, 1), height / max(img.height, 1))
        nw = max(width, int(round(img.width * scale)))
        nh = max(height, int(round(img.height * scale)))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        x0 = max(0, (nw - width) // 2)
        y0 = max(0, (nh - height) // 2)
        img = img.crop((x0, y0, x0 + width, y0 + height))
        return img.convert("RGBA")
    if style == "纯白":
        return Image.new("RGBA", (width, height), (255, 255, 255, 255))
    if style == "蓝白渐变":
        return _gradient_v(
            width,
            height,
            [
                np.array([236, 248, 255], dtype=np.float32),
                np.array([168, 214, 255], dtype=np.float32),
                np.array([96, 176, 246], dtype=np.float32),
            ],
        )
    return _gradient_v(
        width,
        height,
        [
            np.array([214, 186, 248], dtype=np.float32),
            np.array([186, 198, 250], dtype=np.float32),
            np.array([126, 196, 250], dtype=np.float32),
        ],
    )


def draw_footer(canvas, text, footer_h):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or footer_h <= 0:
        return
    w, h = canvas.size
    bar = _gradient_h(w, footer_h, _FOOTER_C0, _FOOTER_C1)
    canvas.alpha_composite(bar, dest=(0, h - footer_h))
    draw = ImageDraw.Draw(canvas)
    size = max(18, int(footer_h * 0.28))
    font = _find_font(size, bold=True)
    while size >= 14:
        font = _find_font(size, bold=True)
        widths = []
        heights = []
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        if max(widths) <= w * 0.92:
            break
        size -= 2
    line_gap = max(4, int(size * 0.22))
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = h - footer_h + (footer_h - total_h) / 2.0
    for ln, tw, th in zip(lines, widths, heights):
        x = (w - tw) / 2.0
        draw.text((x, y), ln, font=font, fill=(255, 255, 255, 255))
        y += th + line_gap


def draw_qty_label(canvas, qty, region):
    x, y, rw, rh = region
    text = f"x{int(qty)}"
    size = int(min(rw, rh) * 0.26)
    size = max(26, min(size, int(min(canvas.size) * 0.085)))
    font = _find_font(size, black=True)
    draw = ImageDraw.Draw(canvas)
    stroke = max(2, size // 18)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x + (rw - tw) / 2.0 - bbox[0]
    ty = y + (rh - th) / 2.0 - bbox[1]
    draw.text(
        (tx, ty),
        text,
        font=font,
        fill=_LABEL_FILL,
        stroke_width=stroke,
        stroke_fill=_LABEL_STROKE,
    )


def compose_kit(slots, width, height, footer_text, bg_style, background, show_shadow):
    footer_text = footer_text or ""
    lines = [ln.strip() for ln in footer_text.splitlines() if ln.strip()]
    footer_h = 0 if not lines else max(72, int(height * 0.125))
    canvas = make_background(width, height, bg_style, background)
    regions = _pack_regions(slots, width, height, footer_h)
    for slot, region in zip(slots, regions):
        draw_stack(
            canvas,
            slot["sprite"],
            slot["copies"],
            slot["mode"],
            region,
            show_shadow,
        )
    for slot, region in zip(slots, regions):
        if slot["label"]:
            draw_qty_label(canvas, slot["qty"], region)
    draw_footer(canvas, footer_text, footer_h)
    return canvas.convert("RGB")


class ProductKitLayout:
    @classmethod
    def INPUT_TYPES(s):
        required = {
            "width": ("INT", {"default": 1536, "min": 512, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 1536, "min": 512, "max": 2048, "step": 16}),
            "footer_text": (
                "STRING",
                {
                    "multiline": True,
                    "default": (
                        "For Dreame L10s Pro Ultra Heat / Mova E30 Ultra\n"
                        "For Dreame D20 Ultra / Dreame L10s Ultra Gen 2"
                    ),
                },
            ),
            "bg_style": (BG_STYLES, {"default": "紫蓝渐变"}),
            "cutout": (CUTOUT_MODES, {"default": "遮罩前景"}),
            "max_copies": ("INT", {"default": 20, "min": 1, "max": 40}),
            "show_shadow": ("BOOLEAN", {"default": True}),
        }
        optional = {
            "bg_removal_model": ("BACKGROUND_REMOVAL",),
            "background": ("IMAGE",),
        }
        for i in range(1, SLOT_COUNT + 1):
            optional[f"part_{i}"] = ("KIT_PART",)
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compose"
    CATEGORY = "image"
    DESCRIPTION = "Composite up to 25 product parts, sized by quantity and stack, without overflowing the poster."

    def compose(
        self,
        width,
        height,
        footer_text,
        bg_style,
        cutout,
        max_copies,
        show_shadow,
        **kwargs
    ):
        bg_model = kwargs.get("bg_removal_model")
        slots = []
        for i in range(1, SLOT_COUNT + 1):
            part = kwargs.get(f"part_{i}")
            if not isinstance(part, dict):
                continue
            image = part.get("image")
            qty = coerce_qty(part.get("qty"))
            if image is None or is_placeholder_image(image) or qty <= 0:
                continue
            mask = part.get("mask")
            if (
                cutout == "遮罩前景"
                and bg_model is not None
                and hasattr(bg_model, "encode_image")
            ):
                try:
                    mask = bg_model.encode_image(image)
                    sprite = tensor_to_rgba(image, mask, "遮罩前景")
                except Exception:
                    sprite = tensor_to_rgba(image, mask, cutout)
            else:
                sprite = tensor_to_rgba(image, mask, cutout)
            sprite = feather_alpha(sprite, 1.0)
            sprite = trim_rgba(sprite)
            longest = max(sprite.size)
            if longest > 1024:
                s = 1024 / longest
                sprite = _scale_rgba(sprite, sprite.width * s, sprite.height * s)
            stack = part.get("stack", "横排层叠")
            if stack not in STACK_MODES:
                stack = "横排层叠"
            copies = min(qty, int(max_copies))
            if stack == "并排":
                copies = min(copies, 2)
            slots.append(
                {
                    "sprite": sprite,
                    "qty": qty,
                    "copies": max(1, copies),
                    "mode": stack,
                    "label": bool(part.get("label", True)),
                }
            )
        if not slots:
            raise ValueError("请至少连接一张配件图，并把对应数量设为大于 0")

        background = kwargs.get("background")
        if is_placeholder_image(background):
            background = None
        if background is not None:
            bg_style = "自定义底图"
        rgb = compose_kit(
            slots,
            int(width),
            int(height),
            footer_text,
            bg_style,
            background,
            bool(show_shadow),
        )
        arr = np.array(rgb).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return (tensor,)


class LoadImageOptional:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.isdir(input_dir):
            files = [
                f
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
            ]
            try:
                files = folder_paths.filter_files_content_types(files, ["image"])
            except Exception:
                pass
        choices = [NONE_CHOICE] + sorted(files)
        return {
            "required": {
                "image": (choices, {"image_upload": True, "default": NONE_CHOICE}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image"
    DESCRIPTION = "Load an image, or leave as （不选）. Empty slots are skipped."

    def load_image(self, image, qty=None):
        if _is_none_choice(image) or not folder_paths.exists_annotated_filepath(image):
            return empty_image_mask()
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
        rgb = img.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,]
        if "A" in img.getbands():
            mask = 1.0 - torch.from_numpy(
                np.array(img.getchannel("A")).astype(np.float32) / 255.0
            )
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32)
        return (tensor, mask.unsqueeze(0))

    @classmethod
    def IS_CHANGED(s, image, qty=None):
        if _is_none_choice(image) or not folder_paths.exists_annotated_filepath(image):
            return (NONE_CHOICE, qty)
        image_path = folder_paths.get_annotated_filepath(image)
        digest = hashlib.sha256()
        with open(image_path, "rb") as f:
            digest.update(f.read())
        return (digest.hexdigest(), qty)

    @classmethod
    def VALIDATE_INPUTS(s, image, qty=None, **kwargs):
        if _is_none_choice(image):
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class LoadProductPart(LoadImageOptional):
    @classmethod
    def INPUT_TYPES(s):
        parent = super().INPUT_TYPES()
        required = dict(parent["required"])
        required["qty"] = (
            "STRING",
            {
                "default": "0",
                "multiline": False,
                "tooltip": "叠放数量，如 10。不选图片时可留空或 0。",
            },
        )
        required["stack"] = (STACK_MODES, {"default": "横排层叠"})
        required["label"] = ("BOOLEAN", {"default": True})
        return {"required": required}

    RETURN_TYPES = ("KIT_PART",)
    RETURN_NAMES = ("part",)
    FUNCTION = "load_part"
    CATEGORY = "image"
    DESCRIPTION = "Optional product-part image, quantity, stack mode, and xN label."

    def load_part(self, image, qty="0", stack="横排层叠", label=True):
        loaded_image, mask = self.load_image(image)
        n = coerce_qty(qty)
        if is_placeholder_image(loaded_image):
            n = 0
        if stack not in STACK_MODES:
            stack = "横排层叠"
        return (
            {
                "image": loaded_image,
                "mask": mask,
                "qty": n,
                "stack": stack,
                "label": bool(label),
            },
        )

    @classmethod
    def IS_CHANGED(s, image, qty=None, stack=None, label=None):
        return (LoadImageOptional.IS_CHANGED(image, qty), stack, label)


class RemoveBackgroundOptional:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bg_removal_model": ("BACKGROUND_REMOVAL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "remove"
    CATEGORY = "image"
    DESCRIPTION = "Foreground mask. Skips the model when the image slot is empty."

    def remove(self, bg_removal_model, image):
        if is_placeholder_image(image):
            batch = int(image.shape[0]) if image.ndim == 4 else 1
            return (
                torch.zeros(
                    (batch, int(image.shape[-3]), int(image.shape[-2])),
                    dtype=image.dtype,
                    device=image.device,
                ),
            )
        mask = bg_removal_model.encode_image(image)
        return (mask,)


NODE_CLASS_MAPPINGS = {
    "ProductKitLayout": ProductKitLayout,
    "LoadImageOptional": LoadImageOptional,
    "LoadProductPart": LoadProductPart,
    "RemoveBackgroundOptional": RemoveBackgroundOptional,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductKitLayout": "Product Kit Layout 配件套装拼版",
    "LoadImageOptional": "Load Image Optional 可选加载",
    "LoadProductPart": "Load Product Part 配件图+数量",
    "RemoveBackgroundOptional": "Remove Background Optional 可选抠图",
}
