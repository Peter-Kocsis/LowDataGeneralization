import logging
import os
import subprocess
import sys
import time

from gui.adl4cv_app.tools.common.tensorboard_view import TensorBoardView
from gui.adl4cv_app.utils import ADL4CVTool


class TensorBoardProcess:
    def __init__(self, log_dir: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_dir = log_dir

        self.process = None

    def start(self):
        self.logger.info("TensorBoard is starting!")
        env = {'SYSTEMROOT': os.environ['SYSTEMROOT']} if sys.platform.lower().startswith('win') else {}
        tensorboard = os.path.join(sys.exec_prefix, "bin", "tensorboard")
        self.process = subprocess.Popen([tensorboard, '--logdir', self.log_dir, '--port', '6006'], env=env)
        time.sleep(5)
        pass

    def stop(self):
        self.logger.info("TensorBoard is shutting down!")
        if self.process and self.process.returncode is None:
            self.process.terminate()
            self.process.wait()


class TensorBoardTool(ADL4CVTool):
    ID = "tensorboard"
    NAME = "TensorBoard"

    def __init__(self, q):
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(q)

        self.tensorboard_process = None

    @property
    def root_view(self):
        if self._root_view is None:
            self._root_view = TensorBoardView(None, self.ID)
        return self._root_view

    def start(self, tool_attribs, view_attribs):
        self.tensorboard_process = TensorBoardProcess(os.path.abspath("logs"))
        self.tensorboard_process.start()

        self.current_view = self.root_view
        self.current_view.show(self.q.value)

    def stop(self):
        if self.tensorboard_process is not None:
            self.tensorboard_process.stop()
            self.tensorboard_process = None

        super().stop()
