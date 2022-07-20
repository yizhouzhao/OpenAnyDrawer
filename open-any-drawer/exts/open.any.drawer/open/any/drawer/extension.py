import omni.ext
import omni.ui as ui

from env import OpenEnv



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
                ui.Button("Add Object", clicked_fn= self.env.add_object)

    def on_shutdown(self):
        print("[open.any.drawer] MyExtension shutdown")

        
