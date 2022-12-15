import logging
import os
from typing import Optional

from lowdataregime.parameters.log_loader import ActiveLearningLogLoader
from gui.adl4cv_app.utils import ADL4CVTool
from gui.adl4cv_app.tools.inspect_logs.log_visualizer import RootScopeView

LOGS_ROOT = os.path.abspath("logs")

class InspectActiveLearningLogsTool(ADL4CVTool):
    ID = "inspect_logs"
    NAME = "Active Learning Logs"

    def __init__(self, q, main_scope: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(q)
        self.main_scope = main_scope
        self.ID = f"{self.ID}_{self.main_scope}"
        self.NAME = f"Logs: {self.main_scope}"

        self._log_loader: Optional[ActiveLearningLogLoader] = None

    @property
    def root_view(self):
        if self._root_view is None:
            self._root_view = RootScopeView(None, self.ID, self.log_loader)
        return self._root_view

    @property
    def log_loader(self):
        if self._log_loader is None:
            self._log_loader = ActiveLearningLogLoader(LOGS_ROOT, self.main_scope)
        return self._log_loader

    async def start(self, tool_attribs, view_attribs):
        self.logger.debug(f"Tool attributes: {tool_attribs}")
        self.logger.debug(f"View attributes: {view_attribs}")

        self.current_view = self.root_view.from_attributes(self.log_loader, tool_attribs, view_attribs,)
        await self.current_view.show(self.q.value)
