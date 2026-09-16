import os
from datetime import datetime

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import ComfyExtension, Input, Types, io, ui
from typing_extensions import override


class SaveVideoTimestamp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideoTimestamp",
            display_name="Save Video (Timestamp)",
            category="video",
            description="Saves videos as prefix_YYYYMMDD_HHMMSS.mp4 without 00001 counters.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="PersonSwap"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Video.Output("video")],
        )

    @staticmethod
    def _unique_path(folder, filename):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file = f"{filename}_{ts}.mp4"
        full = os.path.join(folder, file)
        if not os.path.exists(full):
            return file, full
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file = f"{filename}_{ts}.mp4"
        full = os.path.join(folder, file)
        while os.path.exists(full):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            file = f"{filename}_{ts}.mp4"
            full = os.path.join(folder, file)
        return file, full

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix="PersonSwap") -> io.NodeOutput:
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(folder_paths.get_output_directory(), subfolder)
        os.makedirs(full_output_folder, exist_ok=True)

        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata

        file, path = cls._unique_path(full_output_folder, filename)
        video.save_to(
            path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
            metadata=saved_metadata,
            crf=18,
        )
        return io.NodeOutput(
            video,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )


class SaveVideoTimestampExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveVideoTimestamp]


async def comfy_entrypoint() -> SaveVideoTimestampExtension:
    return SaveVideoTimestampExtension()
