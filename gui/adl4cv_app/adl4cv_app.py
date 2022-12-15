import asyncio
import logging
import os
import warnings

from h2o_wave import app, Q, ui, main

from gui.adl4cv_app.tools.tensorboard.tensorboard import TensorBoardTool
from gui.adl4cv_app.utils import Atomic
from gui.adl4cv_app.tools.inspect_logs.inspect_logs import InspectActiveLearningLogsTool, LOGS_ROOT

app_title = 'ADL4CV'


class ADL4CVApp:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)

        self.q = Atomic(None)
        self._tools = None

        self.current_tool = None
        self.tool_gui = None

    @property
    def features(self):
        folders_to_exclude = ["experiment_status", "optimization"]
        log_scopes = sorted((item for item in os.listdir(LOGS_ROOT)
                      if os.path.isdir(os.path.join(LOGS_ROOT, item)) and
                      item not in folders_to_exclude))

        if self._tools is None and self.q is not None:
            tools = [InspectActiveLearningLogsTool(self.q, log_scope)
                for log_scope in log_scopes] \
                    + [TensorBoardTool(self.q)]
            self._tools = {tool.ID: tool for tool in tools}
            # self._tools = {
            #     InspectActiveLearningLogsTool.ID: InspectActiveLearningLogsTool(self.q, "active_learning"),
            #     TensorBoardTool.ID: TensorBoardTool(self.q),
            #     # ADD New Tools HERE
            # }
        return self._tools

    async def serve(self, q: Q):
        self.q.value = q
        if not self.q.value.client.initialized:
            print("Initializing the client!")
            self.q.value.client.initialized = True
            await self.setup_page()

        # WORKAROUND Clickable table
        if self.q.value.args.gui_active_learning_accuracy is not None:
            print(f"Selecting from table: {self.q.value.args.gui_active_learning_accuracy}")
            selected_feature_key = self.current_tool.route_attribs[1:] + "/" + str(self.q.value.args.gui_active_learning_accuracy)[2:-2]
        else:
            selected_feature_key = self.q.value.args['#']
            if selected_feature_key is None:
                selected_feature_key = list(self.features.keys())[0]
        await self.show_feature(selected_feature_key)

    async def setup_page(self):
        self.q.value.page['meta'] = ui.meta_card(
            box='',
            title=app_title
        )

        self.q.value.page['header'] = ui.header_card(
            box='1 1 2 1',
            title=app_title,
            subtitle=f'Data analytics',
        )

        self.q.value.page['available_features'] = ui.nav_card(
            box='1 2 2 9',
            items=[
                ui.nav_group(
                    label='Analysis tools',
                    items=[ui.nav_item(name=f'#{features_key}', label=features_value.NAME) for
                           features_key, features_value in self.features.items()]
                ),
            ],
        )

        async def async_save_page(value):
            await value.page.save()

        await self.q.async_run(async_save_page)

    async def show_feature(self, selected_tool_key):
        if '/' in selected_tool_key:
            tool_key_parts = selected_tool_key.split('/')
            tool_id = tool_key_parts[0]
            tool_attribs = tool_key_parts[1:]
        else:
            tool_id = selected_tool_key
            tool_attribs = []

        print(f"Tool ID: {tool_id}")

        if len(tool_attribs) != 0:
            attrib_split = tool_attribs[-1].split('#')
            view_attribs = attrib_split[1:]
            tool_attribs[-1] = attrib_split[0]
        else:
            id_split = tool_id.split('#')
            view_attribs = id_split[1:]
            tool_id = id_split[0]

        # Stop active example, if any.
        print(f"Showing feature {tool_id}")
        if self.current_tool:
            print(f"Current feature {self.current_tool} will be stopped")
            self.current_tool.stop()
            print(f"Current feature {self.current_tool} is stopped")

        # Start new example
        self.current_tool = self.features[tool_id]

        event_loop = asyncio.get_running_loop()
        await self.current_tool.start(tool_attribs, view_attribs)
        print(f"Feature {selected_tool_key} started")

        async def async_save_page(value):
            asyncio.ensure_future(value.page.save(), loop=event_loop)

        await self.q.async_run(async_save_page)


adl4cv_app = ADL4CVApp()


@app('/adl4cv')
async def serve(q: Q):
    await adl4cv_app.serve(q)


print('----------------------------------------')
print(' Welcome to the Active Learning Analytics system!')
print('')
print(' Go to http://localhost:10101/adl4cv')
print('----------------------------------------')
