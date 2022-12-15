import logging
import os
import signal
import time
import traceback
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Union, Type, Optional, Mapping

from lowdataregime.classification.log.logger import TENSORBOARD_LOG_DIR, init_logger
from lowdataregime.utils.utils import Serializable
import sys


class ExperimentInfo(ABC):
    @classmethod
    @abstractmethod
    def from_str(cls, info: str):
        raise NotImplementedError()

    @abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError()


class Experiment(ABC):
    EXPERIMENT_QUEUE_PATH = 'experiment_status'

    def __init__(self, arguments: Namespace, job_id: Optional[int] = None, logging_root: str = TENSORBOARD_LOG_DIR):
        self.arguments = arguments
        self.module_logger = init_logger(self.__class__.__name__, logging_level=logging.INFO)

        self.job_id = self._get_job_id(job_id)
        self.logging_root = logging_root

        self._experiment_status_path = os.path.join(self.logging_root,
                                                    self.EXPERIMENT_QUEUE_PATH,
                                                    self._get_status_file_name(self.job_id))

    def _get_job_id(self, job_id: Optional[int] = None):
        if job_id is None:
            if "SLURM_JOB_ID" in os.environ:
                return int(os.environ["SLURM_JOB_ID"])
            else:
                self.module_logger.debug(f"SLURM_JOB_ID not found in {os.environ}")
                self.module_logger.warning("Job ID not found, using default value: -1")
                return -1
        else:
            return job_id

    @staticmethod
    def _get_status_file_name(job_id: Optional[int] = None):
        return f"{job_id}.status"

    @classmethod
    @abstractmethod
    def add_argparse_args(cls, parent_parser=ArgumentParser(add_help=False)):
        """
        Defines the arguments of the class
        :param parent_parser: The parser which should be extended
        :return: The extended parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "-jid", "--job_id", type=int, default=None, help="The ID of the experiment"
        )
        parser.add_argument(
            "-lr", "--logging_root", type=str, default=TENSORBOARD_LOG_DIR, help="Log directory root"
        )
        parser.add_argument(
            "-sss", "--skip_status_saving", action="store_true", default=False,
            help="Skip the status saving"
        )
        return parser

    @classmethod
    def argparser(cls):
        return cls.add_argparse_args()

    @classmethod
    def from_argparse_args(cls: Type['_T'], arguments: Namespace) -> '_T':
        """
        Creates new object from arguments
        """
        return cls(arguments, arguments.job_id, arguments.logging_root)

    def run(self):
        # Handle the job logging and start the job
        max_restarts = 1
        restart_counter = 0
        while restart_counter < max_restarts:
            try:
                experiment_info_str = self._find_experiment_info_str()
                self.prepare(experiment_info_str)
                self.write_experiment_info()
                self.execute()
                self._remove_experiment_info()
                return
            except KeyboardInterrupt:
                self.write_experiment_info()
                self.module_logger.info("Experiment interrupted!")
            except Exception as exp:
                restart_counter += 1
                self.module_logger.fatal(f"Unhandled exception occured during experiment, trying to continue! "
                                         f"\nDetails: {exp}"
                                         f"\nStack trace: {traceback.format_stack()}")
                if restart_counter == max_restarts:
                    raise exp

    @abstractmethod
    def prepare(self, experiment_info_str: Optional[str]):
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        # Execute the job
        raise NotImplementedError()

    def _remove_experiment_info(self):
        if self.arguments.skip_status_saving:
            return

        if os.path.exists(self._experiment_status_path):
            os.remove(self._experiment_status_path)
            self.module_logger.debug(f"Status file {self._experiment_status_path} removed!")
        else:
            self.module_logger.error(f"Unable to remove experiment status file, "
                                     f"{self._experiment_status_path} not found!")

    def _find_experiment_info_str(self):
        if os.path.exists(self._experiment_status_path):
            self.module_logger.debug(f"Status file {self._experiment_status_path} found!")
            with open(self._experiment_status_path, mode='r') as status_file:
                status = status_file.read()
            self.module_logger.debug(f"Status: {status}")
            return status
        else:
            self.module_logger.debug(f"Status file {self._experiment_status_path} not found!")
            return None

    def write_experiment_info(self):
        if self.arguments.skip_status_saving:
            return

        if os.path.exists(self._experiment_status_path):
            is_updated = True
            os.remove(self._experiment_status_path)
        else:
            os.makedirs(os.path.dirname(self._experiment_status_path), exist_ok=True)
            is_updated = False

        status = self.experiment_info().to_str()
        with open(self._experiment_status_path, mode='w') as status_file:
            status_file.write(status)

        if is_updated:
            self.module_logger.info(f"Status file {self._experiment_status_path} updated!")
        else:
            self.module_logger.info(f"Status file {self._experiment_status_path} created!")
        self.module_logger.debug(f"Status: {status}!")

    @abstractmethod
    def experiment_info(self) -> ExperimentInfo:
        raise NotImplementedError()
