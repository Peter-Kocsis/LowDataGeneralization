import logging
import os
import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional, Dict, Union

import pandas as pd

from lowdataregime.classification.data.cifar import CIFARDataModule
from lowdataregime.classification.data.classification_datamodule import ClassificationDataModule
from lowdataregime.active_learning.active_datamodule import ActiveDataModuleStatus
from lowdataregime.active_learning.active_trainer import ActiveTrainer, ActiveTrainerState, ActiveTrainerHyperParameterSet
from lowdataregime.experiment.experiment import Experiment
from lowdataregime.utils.utils import Serializable


class ActiveLearningScope:
    def __init__(self, data_id: str = None, model_id: str = None):
        self.data_id = data_id
        self.model_id = model_id


class ActiveLearningLogLoader:

    def __init__(self, root_folder: str, main_scope: str = ActiveTrainer.DEFAULT_SCOPE):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root_folder = root_folder
        self.main_scope = main_scope

        self._experiment_log_folder: Optional[str] = None
        self._active_learning_log_folder: Optional[str] = None

        self._experiment_logs: Optional[Dict[str, ExperimentLog]] = None
        self._active_learning_logs: Optional[Dict[str, ActiveLearningLog]] = None

    @property
    def experiment_log_folder(self):
        if self._experiment_log_folder is None:
            self._experiment_log_folder = os.path.join(self.root_folder, Experiment.EXPERIMENT_QUEUE_PATH)
        return self._experiment_log_folder

    @property
    def active_learning_log_folder(self):
        if self._active_learning_log_folder is None:
            self._active_learning_log_folder = os.path.join(self.root_folder, self.main_scope)
        return self._active_learning_log_folder

    @property
    def experiment_logs(self):
        if self._experiment_logs is None:
            self.logger.info(f"Searching for experiment logs in folder {self.experiment_log_folder}")

            log_files = pathlib.Path(self.experiment_log_folder).glob('*.log')
            experiment_logs = (ExperimentLog(str(log_file)) for log_file in log_files)
            self._experiment_logs = {experiment_log.job_id: experiment_log for experiment_log in experiment_logs}

            self.logger.debug(f"{len(self._experiment_logs)} experiment logs have been found!")
        return self._experiment_logs

    def get_scope_logs(self, scope: ActiveLearningScope) -> 'ActiveLearningScopeLog':
        scope_logs = self.active_learning_logs
        if scope.data_id is not None:
            scope_logs = {log_id: active_learning_log
                          for log_id, active_learning_log
                          in scope_logs.items()
                          if active_learning_log.data_id == scope.data_id}
        if scope.model_id is not None:
            scope_logs = {log_id: active_learning_log
                          for log_id, active_learning_log
                          in scope_logs.items()
                          if active_learning_log.model_id == scope.model_id}
        return ActiveLearningScopeLog(scope_logs)

    @property
    def active_learning_logs(self):
        if self._active_learning_logs is None:
            self.logger.info(f"Searching for active learning logs in folder {self.active_learning_log_folder}")

            param_files = pathlib.Path(self.active_learning_log_folder).glob(f'**/{ActiveTrainer.PARAMS_FILE}')
            active_learning_logs = (ActiveLearningLog.from_param_file(str(param_file))
                                    for param_file in param_files)
            self._active_learning_logs = {active_learning_log.log_id: active_learning_log
                                          for active_learning_log in active_learning_logs}

            self.logger.debug(f"{len(self._active_learning_logs)} active learning logs have been found!")

        return self._active_learning_logs

    @property
    def model_ids(self) -> List[str]:
        return list({active_learning_log.model_id for active_learning_log in self.active_learning_logs.values()})

    @property
    def data_ids(self) -> List[str]:
        return list({active_learning_log.data_id for active_learning_log in self.active_learning_logs.values()})

    @property
    def log_ids(self) -> List[str]:
        return list(sorted({log_id for log_id in self.active_learning_logs.keys()}))


class ExperimentLog:
    # TODO: Implement more details if needed
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.status_file = self._get_status_file()
        self.job_id = self._get_job_id()

    def _get_status_file(self):
        status_path = os.path.join(os.path.dirname(self.log_file), f"{self.job_id}.status")
        if os.path.exists(status_path):
            return status_path
        return None

    def _get_job_id(self):
        return sum(os.path.basename(os.path.splitext(self.log_file)[0]).split('-'))


class ActiveLearningLog(ABC):
    def __init__(self, log_folder: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_folder = log_folder

        self._model_id = None
        self._data_id = None
        self._log_id = None

        self._param_file: Optional[str] = None
        self._params: Optional[Serializable] = None

        self._datamodule: Optional[ClassificationDataModule] = None

    @classmethod
    def from_param_file(cls, param_file: str):
        if "benchmark" in os.path.basename(os.path.dirname(param_file)):
            return ActiveLearningBenchmarkLog(os.path.dirname(param_file))
        else:
            return ActiveLearningTrainingLog(os.path.dirname(param_file))

    @property
    def datamodule(self) -> CIFARDataModule:
        if self._datamodule is None:
            self._datamodule = self.params.optimization_definition_set.data_definition.instantiate()
        return self._datamodule

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self._model_id = str(self.log_folder.split(os.path.sep)[-2])
        return self._model_id

    @property
    def data_id(self):
        if self._data_id is None:
            self._data_id = str(self.log_folder.split(os.path.sep)[-3])
        return self._data_id

    @property
    def log_id(self):
        if self._log_id is None:
            self._log_id = str(self.log_folder.split(os.path.sep)[-1])
        return self._log_id

    @property
    @abstractmethod
    def results(self) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    @abstractmethod
    def sample_occurences(self) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def param_file(self) -> str:
        if self._param_file is None:
            self._param_file = os.path.join(self.log_folder, ActiveTrainer.PARAMS_FILE)
        return self._param_file

    @property
    def params(self) -> ActiveTrainerHyperParameterSet:
        if self._params is None:
            self._params = ActiveTrainerHyperParameterSet.loads_from_file(self.param_file)
        return self._params

    def get_accuracy_columns(self, results: Optional[pd.DataFrame] = None):
        if results is None:
            results = self.results
        columns_to_skip = ["stage_id", "size_of_labeled_pool"]
        pattern = "accuracy"
        return [column for column in results.columns if column not in columns_to_skip and pattern in column]

    @property
    def params_markdown(self):
        params: ActiveTrainerHyperParameterSet = self.params
        if params is None:
            return '=' \
                   '# No params have been found!'

        return f'=' \
               f'#### Active learning definition\n' \
               f'* Number of training stages: {params.num_train_stages}\n' \
               f'* Number of new labeled samples per stage: {params.num_new_labeled_samples_per_stage}\n' \
               f'* Number of samples to evaluate per stage: {params.num_samples_to_evaluate_per_stage}\n' \
               f'* Train from scratch in each stage: {params.train_from_scratch}\n' \
               f'#### Data definition\n' \
               f'* Type: {params.optimization_definition_set.data_definition.type}\n' \
               f'* Validation ratio: {params.optimization_definition_set.data_definition.hyperparams.val_ratio}\n' \
               f'* Batch size: {params.optimization_definition_set.data_definition.hyperparams.batch_size}\n' \
               f'* Training transforms: {params.optimization_definition_set.data_definition.hyperparams.train_transforms_def}\n' \
               f'* Validation transforms: {params.optimization_definition_set.data_definition.hyperparams.val_transforms_def}\n' \
               f'* Test transforms: {params.optimization_definition_set.data_definition.hyperparams.test_transforms_def}\n' \
               f'#### Model definition\n' \
               f'* Type: {params.optimization_definition_set.model_definition.type}\n' \
               f'* Hyperparams: {params.optimization_definition_set.model_definition.hyperparams}\n' \
               f'#### Trainer definition\n' \
               f'* Max epochs: {params.optimization_definition_set.trainer_definition.hyperparams.max_epochs}\n'

    def __str__(self):
        return f"{self.model_id}/{self.data_id}/{self.log_id}"


class ActiveLearningScopeLog:

    def __init__(self, scope_logs: Dict[str, ActiveLearningLog]):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.scope_logs = scope_logs
        self._total_data_distribution = None
        self._total_test_distribution = None
        self._results: Optional[pd.DataFrame] = None
        self._active_learning_logs: Optional[Dict[str, ActiveLearningLog]] = None

    @property
    def results(self):
        if self._results is None:
            common_columns = ['size_of_labeled_pool']
            self._results = pd.DataFrame(columns=common_columns)
            for scope_log in self.scope_logs.values():
                results = scope_log.results.copy()
                results = results.drop(['stage_id'], axis=1)
                results.columns = [column if column in common_columns else f"{scope_log.log_id}-{column}"
                                   for column in results.columns]

                self._results = self._results.merge(results,
                                                    on=["size_of_labeled_pool"],
                                                    suffixes=("", f"-{scope_log.log_id}"),
                                                    how="outer")

        return self._results

    @property
    def sample_occurences(self):
        return {scope_id: scope_log.sample_occurences for scope_id, scope_log in self.scope_logs.items()}

    @property
    def model_ids(self) -> List[str]:
        return list({active_learning_log.model_id for active_learning_log in self.scope_logs.values()})

    @property
    def data_ids(self) -> List[str]:
        return list({active_learning_log.data_id for active_learning_log in self.scope_logs.values()})

    @property
    def log_ids(self) -> List[str]:
        return list(sorted({log_id for log_id in self.scope_logs.keys()}))

    def get_columns_by_segment(self, results: Optional[pd.DataFrame] = None, segment: str = "accuracy"):
        if results is None:
            results = self.results
        columns_to_skip = ["stage_id", "size_of_labeled_pool"]
        return [column for column in results.columns if column not in columns_to_skip and segment in column]

    @property
    def total_data_distribution(self):
        if self._total_data_distribution is None:
            self._total_data_distribution = next(iter(self.scope_logs.values())).total_data_distribution
        return self._total_data_distribution

    @property
    def total_test_distribution(self):
        if self._total_test_distribution is None:
            self._total_test_distribution = next(iter(self.scope_logs.values())).total_test_distribution
        return self._total_test_distribution


class ActiveLearningBenchmarkLog(ActiveLearningLog):

    def __init__(self, log_folder: str):
        super().__init__(log_folder)

        self._data_distribution = None
        self._total_data_distribution = None
        self._total_test_distribution = None
        self._run_logs: Optional[Dict[int, ActiveLearningTrainingLog]] = None

        self._benchmark_results: Optional[pd.DataFrame] = None
        self._results: Optional[pd.DataFrame] = None

    @property
    def run_logs(self):
        if self._run_logs is None:
            self.logger.info(f"Searching for run logs in folder {self.log_folder}")

            run_folders = list(pathlib.Path(self.log_folder).glob('run_*'))
            run_logs = (ActiveLearningRunLog(str(run_folder), self) for run_folder in sorted(run_folders))
            self._run_logs = {run_log.run_id: run_log for run_log in run_logs}

            self.logger.debug(f"{len(self._run_logs)} run logs have been found!")
        return self._run_logs

    @property
    def benchmark_results(self):
        if self._benchmark_results is None:
            self._benchmark_results = pd.DataFrame(columns=['stage_id', 'size_of_labeled_pool'])
            for run_log in self.run_logs.values():
                self._benchmark_results = self._benchmark_results.merge(run_log.results,
                                                                        on=["stage_id", "size_of_labeled_pool"],
                                                                        suffixes=("", f"-{run_log.run_id}"),
                                                                        how="outer")
        return self._benchmark_results

    @property
    def results(self):
        if self._results is None:
            self._results = pd.DataFrame()

            columns_to_keep = ["stage_id", "size_of_labeled_pool"]
            self._results[columns_to_keep] = self.benchmark_results[columns_to_keep]

            columns_to_average = defaultdict(list)
            for column in self.benchmark_results.columns:
                if column in columns_to_keep:
                    continue

                averaged_column_name = column.split('-')[0]
                columns_to_average[averaged_column_name].append(column)

            for averaged_column in columns_to_average:
                self._results[averaged_column] =\
                    self.benchmark_results[columns_to_average[averaged_column]].mean(axis=1)
                self._results[f"{averaged_column}_std"] = \
                    self.benchmark_results[columns_to_average[averaged_column]].std(axis=1)
        return self._results

    @property
    def sample_occurences(self):
        run_sample_occurences = pd.concat((run_log.sample_occurences for run_log in self.run_logs.values()))
        self.logger.debug(f"Concatenated occurences: {run_sample_occurences}")
        return run_sample_occurences.groupby(run_sample_occurences.index).mean()

    @property
    def data_distribution(self):
        if self._data_distribution is None:
            self._data_distribution = {run_id: run_log.data_distribution for run_id, run_log in self.run_logs.items()}
        return self._data_distribution

    @property
    def total_data_distribution(self):
        if self._total_data_distribution is None:
            self._total_data_distribution = next(iter(self.run_logs.values())).total_data_distribution
        return self._total_data_distribution

    @property
    def total_test_distribution(self):
        if self._total_test_distribution is None:
            self._total_test_distribution = next(iter(self.run_logs.values())).total_test_distribution
        return self._total_test_distribution


class ActiveLearningTrainingLog(ActiveLearningLog):

    def __init__(self, log_folder: str):
        super().__init__(log_folder)

        self._data_distribution = None
        self._total_data_distribution = None
        self._total_test_distribution = None
        self._stage_logs: Optional[List[ActiveLearningTrainingLog]] = None

        self._benchmark_results: Optional[pd.DataFrame] = None
        self._results: Optional[pd.DataFrame] = None

        self._logs: Optional[str] = None

        self._attention_histogram = None

    @property
    def stage_logs(self):
        if self._stage_logs is None:
            self.logger.info(f"Searching for stage logs in folder {self.log_folder}")

            stage_folders = list(pathlib.Path(self.log_folder).glob('stage_*'))
            stage_logs = (ActiveLearningStageLog(str(stage_folder), self) for stage_folder in stage_folders)
            self._stage_logs = {stage_log.stage_id: stage_log
                                for stage_log in sorted(stage_logs, key=lambda stage_log: stage_log.stage_id)}

            self.logger.debug(f"{len(self._stage_logs)} stage logs have been found!")
        return self._stage_logs

    @property
    def result_file_path(self):
        result_file_name = f"{ActiveTrainer.RESULTS_FILE}.{ActiveTrainer.RESULTS_EXT}"
        return os.path.join(self.log_folder, result_file_name)

    @property
    def results(self):
        if self._results is None:
            if os.path.exists(self.result_file_path):
                self._results = pd.read_csv(self.result_file_path)
            else:
                common_columns = ['stage_id', 'size_of_labeled_pool']
                self._results = pd.DataFrame(columns=common_columns)
        return self._results

    @property
    def sample_occurences(self):
        data = {stage_id: stage_log.sample_occurences for stage_id, stage_log in self.stage_logs.items()}
        self.logger.debug(f"Sample occurences: {data}")
        return pd.DataFrame.from_dict(data)

    @property
    def logs(self):
        if self._logs is None:
            log_file_path = os.path.join(self.log_folder, ActiveTrainer.LOG_FILE)
            with open(log_file_path, "r") as log_file:
                self._logs = log_file.read()
        return self._logs

    @property
    def data_distribution(self):
        if self._data_distribution is None:
            self._data_distribution = {stage_id: stage_log.data_distribution for stage_id, stage_log in self.stage_logs.items()}
        return self._data_distribution

    @property
    def total_data_distribution(self):
        if self._total_data_distribution is None:
            self._total_data_distribution = next(iter(self.stage_logs.values())).total_data_distribution
        return self._total_data_distribution

    @property
    def total_test_distribution(self):
        if self._total_test_distribution is None:
            self._total_test_distribution = next(iter(self.stage_logs.values())).total_test_distribution
        return self._total_test_distribution

    @property
    def attention_histogram(self):
        if self._attention_histogram is None:
            self._attention_histogram = {stage_id: stage_log.attention_histogram for stage_id, stage_log in self.stage_logs.items()}
            self._attention_histogram = {key: value for key, value in self._attention_histogram.items() if value is not None}
        return self._attention_histogram


class ActiveLearningRunLog(ActiveLearningTrainingLog):
    def __init__(self, log_folder: str, parent_log: ActiveLearningBenchmarkLog):
        super().__init__(log_folder)

        self.parent_log = parent_log
        self._run_id: Optional[str] = None
        self._benchmark_folder: Optional[str] = None
        self._data_distribution: Optional[dict] = None
        self._total_data_distribution = None
        self._total_test_distribution = None

        self._attention_histogram = None

    @property
    def datamodule(self) -> CIFARDataModule:
        return self.parent_log.datamodule

    @property
    def param_file(self) -> str:
        if self._param_file is None:
            self._param_file = os.path.join(self.benchmark_folder, ActiveTrainer.PARAMS_FILE)
        return self._param_file

    @property
    def result_file_path(self):
        result_file_name = f"{ActiveTrainer.RESULTS_FILE}_{self.run_id}.{ActiveTrainer.RESULTS_EXT}"
        return os.path.join(self.benchmark_folder, result_file_name)

    @property
    def benchmark_folder(self):
        if self._benchmark_folder is None:
            self._benchmark_folder = os.path.dirname(self.log_folder)
        return self._benchmark_folder

    @property
    def run_id(self):
        if self._run_id is None:
            self._run_id = int(os.path.basename(self.log_folder).split("_")[-1])
        return self._run_id

    @property
    def data_distribution(self):
        if self._data_distribution is None:
            self._data_distribution = {stage_id: stage_log.data_distribution for stage_id, stage_log in self.stage_logs.items()}
        return self._data_distribution

    @property
    def total_data_distribution(self):
        if self._total_data_distribution is None:
            self._total_data_distribution = next(iter(self.stage_logs.values())).total_data_distribution
        return self._total_data_distribution

    @property
    def total_test_distribution(self):
        if self._total_test_distribution is None:
            self._total_test_distribution = next(iter(self.stage_logs.values())).total_test_distribution
        return self._total_test_distribution

    @property
    def attention_histogram(self):
        if self._attention_histogram is None:
            self._attention_histogram = {stage_id: stage_log.attention_histogram for stage_id, stage_log in self.stage_logs.items()}
            self._attention_histogram = {key: value for key, value in self._attention_histogram.items() if value is not None}
        return self._attention_histogram


class ActiveLearningStageLog(ActiveLearningLog):

    def __init__(self, log_folder: str, parent_log: Union[ActiveLearningRunLog, ActiveLearningTrainingLog]):
        super().__init__(log_folder)
        self.parent_log = parent_log

        self._stage_id: Optional[str] = None
        self._data_status: Optional[ActiveDataModuleStatus] = None
        self._phase_logs: Optional[Dict[ActiveTrainerState, str]] = None
        self._data_distribution: Optional[dict] = None
        self._total_data_distribution = None
        self._total_test_distribution = None

        self._attention_histogram = None

    @property
    def datamodule(self) -> CIFARDataModule:
        return self.parent_log.datamodule

    @property
    def labeled_pool_indices(self):
        return self.data_status.labeled_pool_indices

    @property
    def labeled_pool(self):
        return [(labeled_idx, self.datamodule.targets_train[labeled_idx % self.datamodule.original_train_len])
                for labeled_idx in self.labeled_pool_indices]

    @property
    def labeled_pool_by_classes(self):
        labeled_pool_by_classes = {class_name: [] for class_name in self.datamodule.classes}
        for labeled_idx, target in self.labeled_pool:
            labeled_pool_by_classes[self.datamodule.classes[target]].append(labeled_idx)
        return labeled_pool_by_classes

    @property
    def training_pool(self):
        return [(idx, target)
                for idx, target in enumerate(self.datamodule.targets_train)]

    @property
    def test_pool(self):
        return [(idx, target)
                for idx, target in enumerate(self.datamodule.targets_test)]

    @property
    def training_pool_by_classes(self):
        training_pool_by_classes = {class_name: [] for class_name in self.datamodule.classes}
        for train_idx, target in self.training_pool:
            training_pool_by_classes[self.datamodule.classes[target]].append(train_idx)
        return training_pool_by_classes

    @property
    def test_pool_by_classes(self):
        test_pool_by_classes = {class_name: [] for class_name in self.datamodule.classes}
        for train_idx, target in self.test_pool:
            test_pool_by_classes[self.datamodule.classes[target]].append(train_idx)
        return test_pool_by_classes

    @property
    def sample_occurences(self):
        return self.data_status.sample_occurences_in_labeled_pool

    @property
    def data_status(self):
        if self._data_status is None:
            checkpoint_file = os.path.join(self.log_folder, ActiveTrainer.DATA_CHECKPOINT_NAME)
            self._data_status = ActiveDataModuleStatus.loads_from_file(checkpoint_file)
        return self._data_status

    @property
    def phase_logs(self):
        if self._phase_logs is None:
            self._phase_logs = defaultdict(str)
            for phase in ActiveTrainerState:
                phase_log_file = os.path.join(self.log_folder, f"{phase.value}.log")
                if os.path.exists(phase_log_file):
                    with open(phase_log_file, "r") as phase_log:
                        self._phase_logs[phase] = phase_log.read()
        return self._phase_logs

    @property
    def stage_id(self):
        if self._stage_id is None:
            self._stage_id = int(os.path.basename(self.log_folder).split('_')[-1])
        return self._stage_id

    @property
    def results(self) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def data_distribution(self):
        if self._data_distribution is None:
            self._data_distribution = {class_name: len(class_indices) for class_name, class_indices in self.labeled_pool_by_classes.items()}
        return self._data_distribution

    @property
    def total_data_distribution(self):
        if self._total_data_distribution is None:
            self._total_data_distribution = {class_name: len(class_indices) for class_name, class_indices in self.training_pool_by_classes.items()}
        return self._total_data_distribution

    @property
    def total_test_distribution(self):
        if self._total_test_distribution is None:
            self._total_test_distribution = {class_name: len(class_indices) for class_name, class_indices in self.test_pool_by_classes.items()}
        return self._total_test_distribution

    @property
    def attention_histogram(self):
        if self._attention_histogram is None:
            attention_histogram_file = os.path.join(self.log_folder, ActiveTrainer.ATTENTION_HISTOGRAM)
            if os.path.exists(attention_histogram_file):
                self._attention_histogram = pd.read_csv(attention_histogram_file)
        return self._attention_histogram
