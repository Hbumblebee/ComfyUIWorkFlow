import json
import os
from datetime import datetime

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import ComfyExtension, IO, Types
from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes
from typing_extensions import override


class SaveGLBTimestamp(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveGLBTimestamp",
            display_name="Save 3D Model (Timestamp)",
            search_aliases=["export 3d model", "save mesh", "blender", "glb"],
            category="3d",
            description="Saves meshes as prefix_YYYYMMDD_HHMMSS.glb without 00001 counters. Import the GLB into Blender.",
            inputs=[
                IO.Mesh.Input("mesh", tooltip="Mesh to save as GLB."),
                IO.String.Input("filename_prefix", default="Character3D"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @staticmethod
    def _unique_path(folder, filename):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file = f"{filename}_{ts}.glb"
        full = os.path.join(folder, file)
        if not os.path.exists(full):
            return file, full
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file = f"{filename}_{ts}.glb"
        full = os.path.join(folder, file)
        while os.path.exists(full):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            file = f"{filename}_{ts}.glb"
            full = os.path.join(folder, file)
        return file, full

    @classmethod
    def execute(cls, mesh: Types.MESH, filename_prefix: str = "Character3D") -> IO.NodeOutput:
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(folder_paths.get_output_directory(), subfolder)
        os.makedirs(full_output_folder, exist_ok=True)

        metadata = {}
        if not args.disable_metadata:
            if cls.hidden.prompt is not None:
                metadata["prompt"] = json.dumps(cls.hidden.prompt)
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        results = []
        verts = mesh.vertices
        batch = len(verts) if isinstance(verts, list) else verts.shape[0]
        for i in range(batch):
            glb = mesh_item_to_glb_bytes(mesh, i, metadata or None)
            if glb is None:
                continue
            file, path = cls._unique_path(full_output_folder, filename)
            with open(path, "wb") as fh:
                fh.write(glb)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output",
            })
        return IO.NodeOutput(ui={"3d": results})


class SaveGLBTimestampExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SaveGLBTimestamp]


async def comfy_entrypoint() -> SaveGLBTimestampExtension:
    return SaveGLBTimestampExtension()
