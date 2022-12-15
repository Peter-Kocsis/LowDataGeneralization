import logging
import os
from datetime import datetime
from typing import Optional, Union, List

from pytorch_lightning.loggers import TensorBoardLogger

from lowdataregime.parameters.params import OptimizationDefinitionSet

TENSORBOARD_LOG_DIR = os.path.join(".", "logs")


def get_timestamp() -> str:
    """
    Creates timestamp
    :return: Timestamp
    """
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def init_logger(name: str = None, logging_level = logging.DEBUG) -> logging.Logger:
    """
    Creates a new logger with the given name
    :param name: The name of the logger
    :return: The created logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(logging.StreamHandler())
    logger.debug(f"Logger created with name: {name}")
    return logger


class ClassificationLogger(TensorBoardLogger):
    """Logger with DefinitionSet logging and logging handling"""

    def __init__(self,
                 save_dir: str = TENSORBOARD_LOG_DIR,
                 name: Optional[str] = "default",
                 scopes: List[str] = None,
                 version: Optional[Union[int, str]] = None,
                 log_graph: bool = False,
                 default_hp_metric: bool = True, **kwargs):
        """
        Creates new logger instance
        :param save_dir: The root directory of the logs
        :param name: The name of the logs
        :param scopes: List of scope levels. If defined, the parameter <name> is ignored
        :func:`~TensorBoardLogger.__init__`
        """
        if scopes is None:
            folder_name = name
        else:
            folder_name = os.path.join(*scopes, name)
        super().__init__(save_dir, folder_name, version, log_graph, default_hp_metric, **kwargs)

        self.logging_handlers = None

    def start_monitoring(self, log_file_name: str = "logs.log"):
        """
        Initialize the logger to pipe all the logs into a log file
        """
        self.logging_handlers = []
        os.makedirs(self.log_dir, exist_ok=True)
        logging_path = os.path.join(self.log_dir, log_file_name)

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        self.logging_handlers.append(logging.FileHandler(logging_path, mode="w"))
        logger.addHandler(self.logging_handlers[0])
        return logger

    def end_monitoring(self):
        logger = logging.getLogger()
        for handler in self.logging_handlers:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
        self.logging_handlers = None

    def log_parameters(self, optimization_definition_set: OptimizationDefinitionSet):
        """
        Log the parameters to file
        :param optimization_definition_set: The definition set to be logged
        """
        param_file_path = os.path.join(self.log_dir, "params.json")
        optimization_definition_set.dumps_to_file(param_file_path)

