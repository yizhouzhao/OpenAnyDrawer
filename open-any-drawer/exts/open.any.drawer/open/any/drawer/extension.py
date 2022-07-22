import omni.ext
import omni.ui as ui

from .open_env import OpenEnv


# go to directory: open-any-drawer/exts/open.any.drawer/open/any/drawer/
#  # start notebook from: /home/yizhou/.local/share/ov/pkg/isaac_sim-2022.1.0/jupyter_notebook.sh



# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[open.any.drawer] MyExtension startup")

        self.env = OpenEnv()

        self._window = ui.Window("Open any drawer", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
                ui.Button("Add Franka Robot", clicked_fn= self.env.add_robot)

                with ui.HStack(height = 20):
                    ui.Label("object index: ", width = 80)
                    self.object_id_ui = omni.ui.IntField(height=20, width = 40, style={ "margin": 2 })
                    self.object_id_ui.model.set_value(0)
                    ui.Button("Add Object", clicked_fn=self.add_object)

                ui.Button("Add Ground", clicked_fn=self.add_ground)

                ui.Button("Debug", clicked_fn= self.debug)

    def add_ground(self):
        from utils import add_ground_plane

        add_ground_plane("/World/Game")

    def add_object(self):
        object_id = self.object_id_ui.model.get_value_as_int()
        self.env.add_object(object_id)

    def on_shutdown(self):
        print("[open.any.drawer] MyExtension shutdown")

    def debug(self):
        from .utils import get_bounding_box, add_physical_material_to, fix_linear_joint

        # stage = omni.usd.get_context().get_stage()
        # prim = "/World/Game/mobility/link_0/visuals/handle_6"
        # print("get_bounding_box", get_bounding_box(prim))
        b_list = self.env.get_mesh_bboxes("handle_")
        print("bounding box list", b_list)
