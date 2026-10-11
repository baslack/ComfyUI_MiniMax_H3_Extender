"""MiniMax H3 Project: names a clip chain's cache.

Without a project, a chain's cache is named after the node that starts it, so
two workflows with the same node ids share one cache. A project name keeps
each workflow's clips in their own files.
"""
from .motion_context_disk import PROJECT_TYPE, _project_name


class MiniMaxH3Project:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "tooltip": "The cache's name: letters, digits, '-' and '_'. Workflows with the same name share one cache."}),
            },
        }

    RETURN_TYPES = (PROJECT_TYPE,)
    RETURN_NAMES = ("project",)
    FUNCTION = "project"
    CATEGORY = "MiniMax H3"
    DESCRIPTION = "Names the clip chain's cache. Connect it to the chain's first Disk Join, or to Continue Video."

    def project(self, name):
        return (_project_name(name),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Project": MiniMaxH3Project,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Project": "MiniMax H3 Project",
}
