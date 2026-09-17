import logging

import torch

import comfy.model_management
import comfy.samplers
import nodes
from comfy_extras.nodes_scail import WanSCAILToVideo
from nodes import VAEDecode, common_ksampler

logger = logging.getLogger("SCAIL2ChunkLoop")


def _align_4n1(n):
    return max(1, ((int(n) - 1) // 4) * 4 + 1)


def _estimated_chunks(frame_count, chunk_length, overlap):
    first = min(chunk_length, _align_4n1(frame_count))
    if frame_count <= first:
        return 1
    step = max(1, chunk_length - overlap)
    leftover = frame_count - first
    return 1 + (leftover + step - 1) // step


class SCAIL2ChunkLoop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "pose_video": ("IMAGE",),
                "pose_video_mask": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "reference_image_mask": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 896, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chunk_length": ("INT", {"default": 81, "min": 5, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "overlap": ("INT", {"default": 5, "min": 1, "max": 65}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "video"
    DESCRIPTION = (
        "SCAIL-2 replacement in 81-frame chunks with 5-frame overlap, "
        "looping until the driving video ends. Peak VRAM stays one chunk."
    )

    def run(
        self,
        model,
        positive,
        negative,
        vae,
        pose_video,
        pose_video_mask,
        reference_image,
        reference_image_mask,
        width,
        height,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        chunk_length,
        overlap,
        clip_vision_output=None,
    ):
        n_frames = int(pose_video.shape[0])
        if n_frames < 1:
            raise ValueError("SCAIL2ChunkLoop: pose_video is empty.")
        if overlap >= chunk_length:
            raise ValueError("SCAIL2ChunkLoop: overlap must be smaller than chunk_length.")

        n_est = _estimated_chunks(n_frames, chunk_length, overlap)
        decoder = VAEDecode()
        pieces = []
        previous_frames = None
        video_frame_offset = 0
        collected = 0
        chunk_index = 0

        logger.info(
            "SCAIL-2 loop: %s driving frames, ~%s chunks of %s (overlap %s)",
            n_frames,
            n_est,
            chunk_length,
            overlap,
        )

        while collected < n_frames:
            comfy.model_management.throw_exception_if_processing_interrupted()
            chunk_index += 1
            if chunk_index > 1000:
                raise RuntimeError("SCAIL2ChunkLoop: stopped after 1000 chunks.")

            if previous_frames is None:
                length = min(chunk_length, _align_4n1(n_frames))
                offset_in = 0
            else:
                length = chunk_length
                offset_in = video_frame_offset
                adjusted = max(0, offset_in - min(overlap, previous_frames.shape[0]))
                if pose_video.shape[0] <= adjusted:
                    break

            logger.info(
                "SCAIL-2 chunk %s/%s  offset=%s  have=%s/%s",
                chunk_index,
                n_est,
                offset_in,
                collected,
                n_frames,
            )

            packed = WanSCAILToVideo.execute(
                positive,
                negative,
                vae,
                width,
                height,
                length,
                1,
                1.0,
                0.0,
                1.0,
                offset_in,
                overlap,
                replacement_mode=True,
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                pose_video=pose_video,
                pose_video_mask=pose_video_mask,
                reference_image_mask=reference_image_mask,
                previous_frames=previous_frames,
            )
            chunk_positive, chunk_negative, latent, video_frame_offset = packed.args
            sampled = common_ksampler(
                model,
                int(seed) + chunk_index - 1,
                steps,
                cfg,
                sampler_name,
                scheduler,
                chunk_positive,
                chunk_negative,
                latent,
                denoise=denoise,
            )[0]
            decoded = decoder.decode(vae, sampled)[0].to("cpu")
            del sampled, latent, chunk_positive, chunk_negative

            if previous_frames is None:
                pieces.append(decoded)
                collected = decoded.shape[0]
            else:
                extra = decoded[min(overlap, decoded.shape[0]) :]
                if extra.shape[0] == 0:
                    break
                pieces.append(extra)
                collected += extra.shape[0]
            previous_frames = decoded
            comfy.model_management.soft_empty_cache()

        images = torch.cat(pieces, dim=0)
        if images.shape[0] > n_frames:
            images = images[:n_frames]
        logger.info("SCAIL-2 loop done: %s frames in %s chunks", images.shape[0], chunk_index)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "SCAIL2ChunkLoop": SCAIL2ChunkLoop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL2ChunkLoop": "SCAIL-2 Chunk Loop",
}
