'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import glob
import logging
import os
# Torch
import sys
from typing import Optional, Callable

import git
import pandas as pd
import numpy as np
import plotly.express as px
# Custom
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerState

from lowdataregime.classification.data.cifar import CIFAR10Definition, CIFAR10HyperParameterSet
from lowdataregime.classification.log.logger import get_timestamp, init_logger, TENSORBOARD_LOG_DIR, ClassificationLogger
from lowdataregime.classification.model.classification_module import TrainStage
from lowdataregime.classification.model.dummy.fc_classifier import DummyModuleDefinition, \
    DummyModuleHyperParameterSet
from lowdataregime.classification.model.models import ModelType
from lowdataregime.classification.optimizer.optimizers import SGDOptimizerDefinition, \
    SGDOptimizerHyperParameterSet
from lowdataregime.classification.optimizer.schedulers import MultiStepLRSchedulerDefinition, \
    MultiStepLRSchedulerHyperParameterSet
from lowdataregime.parameters.params import OptimizationDefinitionSet, DefinitionSet, \
    HyperParameterSet, Status
from lowdataregime.classification.trainer.trainers import PL_TrainerDefinition, PL_TrainerHyperParameterSet, TrainerType
from lowdataregime.utils.summary import summary
from .active_datamodule import ActiveDataModuleStatus, \
    ActiveDataModuleHyperParameterSet, ActiveDataModuleDefinition
from lowdataregime.experiment.experiment import Experiment
from .query.queries import QueryDefinition, QueryType
from .query.uncertainty_query import UncertaintyQueryDefinition
from lowdataregime.utils.utils import SerializableEnum


class ActiveTrainerType(SerializableEnum):
    """Definition of the implemented active learning trainers"""
    ActiveTrainer = "ActiveTrainer"


class ActiveTrainerState(SerializableEnum):
    PREPARATION = "preparation"
    TRAINING = "training"
    QUERY = "query"
    FINISHED = "finished"


class ActiveTrainerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the ActiveTrainer"""

    def __init__(self,
                 num_train_stages: int = 10,
                 num_initial_samples: int = 1000,
                 num_new_labeled_samples_per_stage: int = 1000,
                 num_samples_to_evaluate_per_stage: int = 10000,
                 train_from_scratch: bool = False,
                 benchmark_id: str = None,
                 run_id: str = None,
                 optimization_definition_set: OptimizationDefinitionSet = None,
                 query_definition_set: QueryDefinition = UncertaintyQueryDefinition(),
                 datamodule_hyperparams: ActiveDataModuleHyperParameterSet = ActiveDataModuleHyperParameterSet(),
                 logging_root=TENSORBOARD_LOG_DIR,
                 output_dir: Optional[str] = None,
                 iid_inference: bool = False,
                 iid_training: bool = False,
                 train_head: bool = False,
                 description: str = None,
                 scope: str = None,
                 keep_checkpoints: bool = False,
                 initial_pool_file: Optional[str] = None,
                 query_ascedning: bool = False,
                 robustness_benchmark: bool = False,
                 **kwargs):
        """
        Creates new HyperParameterSet
        :param num_train_stages: The number of active learning training stages
        :param num_new_labeled_samples_per_stage: The number of unlabeled samples to be labeled on each stage
        :param num_samples_to_evaluate_per_stage: The number of unlabeled samples to be evaluated on each stage
        :param inference_training_sample_ratio: The injection ratio of training samples during uncertainty estimation
        :param temperature: Temperature of uncertainty estimation
        :param train_from_scratch: Indicates, whether to reinitialize the weights of the model on each stage or not
        :param return_logits: Indicates, whether to return the logits or not
        :param return_uncertainties: Indicates, whether to return the uncertainties or not
        :param benchmark_id: The benchmark ID of the run
        :param run_id: The run ID of the run
        :param optimization_definition_set: The definition of the optimization, which will be conducted as active learning
        :param logging_root: The root of the logging folder
        :param random_labeling: Indicates, whether to apply random labeling instead uncertaincy-based method or not
        """
        super().__init__(**kwargs)
        self.num_train_stages = num_train_stages
        self.num_initial_samples = num_initial_samples
        self.num_new_labeled_samples_per_stage = num_new_labeled_samples_per_stage
        self.num_samples_to_evaluate_per_stage = num_samples_to_evaluate_per_stage
        self.train_from_scratch = train_from_scratch
        self.benchmark_id = benchmark_id
        self.run_id = run_id
        self.optimization_definition_set = optimization_definition_set
        self.query_definition = query_definition_set
        self.datamodule_hyperparams = datamodule_hyperparams
        self.logging_root = logging_root
        self.output_dir = output_dir
        self.iid_inference = iid_inference
        self.iid_training = iid_training
        self.train_head = train_head
        self.description = description
        self.scope = scope
        self.keep_checkpoints = keep_checkpoints
        self.initial_pool_file = initial_pool_file
        self.query_ascending = query_ascedning
        self.robustness_benchmark = robustness_benchmark

        self.git_commit_id = git.Repo(search_parent_directories=True).head.object.hexsha
        self.seed = self.get_seed()

    def get_seed(self):
        if self.optimization_definition_set is not None and self.optimization_definition_set.seed is not None:
            seed = self.optimization_definition_set.seed
        else:
            return None

        if self.run_id is not None:
            seed += self.run_id
        return seed


class ActiveTrainerStatus(Status):
    def __init__(self,
                 status_file: str = None,
                 job_id: int = -1,
                 training_folder_path: str = None,
                 current_stage: int = 0,
                 state: ActiveTrainerState = ActiveTrainerState.PREPARATION):
        super().__init__(status_file, job_id)
        self.training_folder_path = training_folder_path
        self.current_stage = current_stage
        self.state = state

    @property
    def stage_id(self):
        return f"stage_{self.current_stage}"

    def get_stage_id(self, current_stage: int):
        return f"stage_{current_stage}"


class ActiveTrainerDefinition(DefinitionSet):
    """Definition of the ActiveTrainer"""

    def __init__(self,
                 hyperparams: ActiveTrainerHyperParameterSet = ActiveTrainerHyperParameterSet(),
                 status: Optional[ActiveTrainerStatus] = None):
        super().__init__(ActiveTrainerType.ActiveTrainer, hyperparams)
        self.status = status

    @property
    def _instantiate_func(self) -> Callable:
        # TODO: Make it cleaner -> maybe remove
        return None

    def instantiate(self, experiment, *args, **kwargs):
        """Instantiates the module"""
        return ActiveTrainer(self.hyperparams, self.status, experiment)


class ActiveTrainer:
    """Class for active learning"""
    MODEL_CHECKPOINT_NAME = "last.ckpt"
    DATA_CHECKPOINT_NAME = "data.json"
    DEFAULT_SCOPE = "active_learning"
    PARAMS_FILE = "active_learning_params.json"
    LOG_FILE = "active_learning.log"

    RESULTS_FILE = "active_learning_results"
    RESULTS_EXT = "csv"

    ATTENTION_HISTOGRAM = "attention_histogram.csv"
    MOST_INFLUENTIAL_SAMPLES = "most_influential_samples.csv"

    def __init__(self,
                 params: ActiveTrainerHyperParameterSet = ActiveTrainerHyperParameterSet(),
                 status: Optional[ActiveTrainerStatus] = None,
                 experiment: Experiment = None):
        assert params.optimization_definition_set is not None, "Optimization is not defined!"
        assert (params.run_id is None) == (
                params.benchmark_id is None), "If benchmark ID is defined, then run ID is required!"

        self.params = params
        self.status = self._get_status(status)
        self.experiment = experiment

        self._benchmark_folder_path = None

        self._training_scope = None
        self._training_name = None
        self._training_result_file = None
        self._training_log_file = None
        self._training_params_file = None

        self._training_results = None

        self._model_checkpoint = None
        self._resume_from_checkpoint = None
        self.model = None
        self.datamodule = None

        self.phase_logger = None

        self.logger = init_logger(self.__class__.__name__)
        self.setup_logger()

        self.inference_labeled_size = None
        self.inference_unlabeled_size = None

    def train(self):
        """
        Runs the active learning training.
        :return: The results of the active learning training
        """
        try:
            self._restore_members()
            while self.status.state != ActiveTrainerState.FINISHED:

                if self.status.state == ActiveTrainerState.PREPARATION:
                    self._prepare()
                    self.status.state = ActiveTrainerState.TRAINING
                elif self.status.state == ActiveTrainerState.TRAINING:
                        self._train_stage()
                        self.status.state = ActiveTrainerState.QUERY
                elif self.status.state == ActiveTrainerState.QUERY:
                    if self.status.current_stage + 1 < self.params.num_train_stages:
                        self.query()
                        self.status.current_stage += 1
                        self.status.state = ActiveTrainerState.TRAINING
                    else:
                        self.status.state = ActiveTrainerState.FINISHED

                self._update_phase_logger()
                self.experiment.write_experiment_info()
            self.clean_up()

        except (KeyboardInterrupt, Exception) as exp:
            self.logger.error(exp)
            if self.phase_logger is not None:
                self.phase_logger.end_monitoring()
            raise
        return self.training_results

    def _restore_members(self):
        self._set_seed()
        if self.status.state == ActiveTrainerState.PREPARATION:
            self.model = self._initialize_model()
            self.datamodule = self._initialize_datamodule()

            self._resume_from_checkpoint = None

            self.phase_logger = self._create_stage_logger(0)
            self.phase_logger.start_monitoring(self._phase_log_path)

        elif self.status.state == ActiveTrainerState.TRAINING:
            self.model = self._restore_model(self.status.current_stage - 1)
            self.datamodule = self._restore_datamodule(self.status.current_stage)

            self._resume_from_checkpoint = \
                self._get_existing_checkpoint(self.status.current_stage, self.MODEL_CHECKPOINT_NAME)

            self.phase_logger = self._create_stage_logger(self.status.current_stage)
            self.phase_logger.start_monitoring(self._phase_log_path)

        elif self.status.state == ActiveTrainerState.QUERY:
            self.model = self._restore_model(self.status.current_stage)
            self.datamodule = self._restore_datamodule(self.status.current_stage)

            self._resume_from_checkpoint = None

            self.phase_logger = self._create_stage_logger(self.status.current_stage)
            self.phase_logger.start_monitoring(self._phase_log_path)

    def _prepare(self):
        if self.params.initial_pool_file is None:
            self.datamodule.label_initial_samples(self.params.num_initial_samples, self.model.feature_model, self.params.optimization_definition_set.trainer_definition.hyperparams.device)
        else:
            self.datamodule.update_train_sampler(self.model.feature_model, self.params.optimization_definition_set.trainer_definition.hyperparams.device)
            self.datamodule.update_test_sampler(self.model.feature_model, self.params.optimization_definition_set.trainer_definition.hyperparams.device)
        self.datamodule.prepare_validation_set()
        self.datamodule.status.dumps_to_file(self._get_checkpoint(self.status.current_stage, self.DATA_CHECKPOINT_NAME))

        self._log_command()
        self._log_params()

        try:
            device = self.params.optimization_definition_set.trainer_definition.hyperparams.device
            if not isinstance(device, str):
                device = device.value

            if self.params.optimization_definition_set.model_definition.type == ModelType.KDCLNet:
                input_size = (self.datamodule.dims, self.datamodule.dims)
            else:
                input_size = self.datamodule.dims

            summary(self.model.to(device),
                    input_size=input_size,
                    batch_size=self.datamodule.batch_size,
                    device=device)
        except Exception as exp:
            self.logger.error(f"Unable to print summary: {exp}")

    @property
    def _is_mpn(self):
        return self.params.optimization_definition_set.model_definition.type in (ModelType.GeneralNet, ModelType.FeatureRefiner)

    @property
    def _is_dml(self):
        return self.params.optimization_definition_set.model_definition.type in (ModelType.DMLNet, ModelType.KDCLNet)

    @property
    def _is_kd(self):
        return self.params.optimization_definition_set.model_definition.type in (ModelType.KDNet,)

    @property
    def _trainer_type(self):
        return self.params.optimization_definition_set.trainer_definition.type

    def _train_stage(self):
        stage_result = dict()

        stage_result["stage_id"] = self.status.stage_id
        stage_result["size_of_labeled_pool"] = self.datamodule.labeled_pool_size
        self.logger.info(
            f"Stage {self.status.current_stage + 1}/{self.params.num_train_stages} || Label set size {self.datamodule.labeled_pool_size} started!")

        if self.params.train_from_scratch:
            self.logger.info("Reinstantiating the model!")
            self.model = self.params.optimization_definition_set.model_definition.instantiate()

        # --------------------------------------------
        # -----------------------------------------------
        # Logging
        # -----------------------------------------------

        callbacks = [ModelCheckpoint(
            dirpath=self.phase_logger.log_dir,
            filename=self.MODEL_CHECKPOINT_NAME,
            save_last=True,
            verbose=True)]

        if self._trainer_type == TrainerType.LearningLossTrainer:
            train_results = self._train_learning_loss(callbacks)
        elif self._trainer_type == TrainerType.FixMatchTrainer:
            train_results = self._train_fixmatch(callbacks)
        else:
            if self._is_dml:
                train_results = self._train_stage_dml(callbacks)
            elif self._is_kd:
                train_results = self._train_stage_kd(callbacks)
            elif self._is_mpn:
                train_results = self._train_stage_mpn(callbacks)
            else:
                train_results = self._train_stage_iid(callbacks)
        stage_result.update(train_results)

        self.add_stage_results(stage_result)
        self.log_results()

    def _train_learning_loss(self, callbacks):
        # --------------------------------------------
        train_results = {}

        # -------------------- LL Training ------------------------
        model_trainer: Trainer = self.params.optimization_definition_set.trainer_definition.instantiate(
            logger=self.phase_logger,
            callbacks=callbacks,
            resume_from_checkpoint=self._resume_from_checkpoint)
        model_trainer.fit(model=self.model, datamodule=self.datamodule.dataset)

        self.logger.info("Model fit done!")

        main_accuracy = model_trainer.test(model=self.model, datamodule=self.datamodule.dataset)
        train_results["main_accuracy"] = main_accuracy

        self.logger.info(f"Stage {self.status.current_stage + 1}/{self.params.num_train_stages} "
                         f"|| Labeled set size {self.datamodule.labeled_pool_size} done: "
                         f"Main model test acc {main_accuracy}")

        return train_results

    def _train_fixmatch(self, callbacks):
        # --------------------------------------------
        train_results = {}

        # -------------------- LL Training ------------------------
        model_trainer: Trainer = self.params.optimization_definition_set.trainer_definition.instantiate(
            logger=self.phase_logger,
            callbacks=callbacks,
            resume_from_checkpoint=self._resume_from_checkpoint)
        model_trainer.fit(model=self.model, datamodule=self.datamodule)

        self.logger.info("Model fit done!")

        main_accuracy = model_trainer.test(model=self.model, datamodule=self.datamodule)
        train_results["main_accuracy"] = main_accuracy

        self.logger.info(f"Stage {self.status.current_stage + 1}/{self.params.num_train_stages} "
                         f"|| Labeled set size {self.datamodule.labeled_pool_size} done: "
                         f"Main model test acc {main_accuracy}")

        return train_results

    def _train_stage_mpn(self, callbacks):
        # --------------------------------------------
        train_results = {}
        if not self.params.iid_training:
            # -------------------- MPN Training ------------------------
            self._train_model(callbacks)

            mpn_results = self._test_model(callbacks)
            train_results.update(mpn_results)

            if self.params.train_head:
                # -------------------- ResNet Head Training  - ResNet Backbone frozen ------------------------
                self._train_iid_head(callbacks)

            iid_results = self._test_iid(callbacks)
            train_results.update(iid_results)
        else:
            self._train_iid(callbacks)

            iid_results = self._test_iid(callbacks)
            train_results.update(iid_results)

        return train_results

    def _train_stage_iid(self, callbacks):
        train_results = {}
        model_trainer = self._train_model(callbacks)

        self.model.init_metrics()

        model_trainer.test(model=self.model, datamodule=self.datamodule.dataset)
        self.logger.info("Model tested!")

        acc = self.model.test_acc.compute().cpu().numpy().tolist()
        acc_5 = self.model.test_5_acc.compute().cpu().numpy().tolist()
        train_results["accuracy"] = acc
        train_results["accuracy_top5"] = acc_5

        self.logger.info(f"Stage {self.status.current_stage + 1}/{self.params.num_train_stages} "
                         f"|| Labeled set size {self.datamodule.labeled_pool_size} done: "
                         f"Model test acc {acc}, top-5: {acc_5}")

        if self.params.robustness_benchmark:
            self.logger.info("Running robustness benchmark!")
            robustness_dataloaders = self.datamodule.dataset.robustness_dataloader()
            for corruption_type, courrupted_dataloaders in robustness_dataloaders.items():
                for courruption_severity, severity_dataloader in courrupted_dataloaders.items():
                    self.model.init_metrics()
                    model_trainer.test(model=self.model, test_dataloaders=[severity_dataloader])
                    acc = self.model.test_acc.compute()
                    train_results[f"accuracy_{corruption_type}_{courruption_severity}"] = acc.cpu().numpy().tolist()
                    self.logger.info(f"{corruption_type} - {courruption_severity}: {acc}")

        return train_results

    def _train_stage_dml(self, callbacks):
        train_results = {}
        model_trainer = self._train_model(callbacks)

        self.model.init_metrics()

        model_trainer.test(model=self.model, datamodule=self.datamodule.dataset)
        self.logger.info("Model tested!")

        acc = self.model.test_acc.compute().cpu().numpy().tolist()
        mutual_acc = self.model.test_mutual_acc.compute().cpu().numpy().tolist()
        train_results["accuracy"] = acc
        train_results["accuracy_mutual"] = mutual_acc

        self.logger.info(f"Stage {self.status.current_stage + 1}/{self.params.num_train_stages} "
                         f"|| Labeled set size {self.datamodule.labeled_pool_size} done: "
                         f"Model test acc {acc}, mutual_acc: {mutual_acc}")
        return train_results

    def _train_stage_kd(self, callbacks):
        train_results = {}

        # Train the teacher model
        full_model = self.model
        self.model = full_model.model.mutual_model
        teacher_trainer = self._train_model(callbacks)

        self.model.init_metrics()

        teacher_trainer.test(model=self.model, datamodule=self.datamodule.dataset)
        self.logger.info("Teacher model tested!")

        teacher_acc = self.model.test_acc.compute().cpu().numpy().tolist()
        train_results["teacher_accuracy"] = teacher_acc
        self.logger.info(f"Teacher accuracy: {teacher_acc}")

        # Train the student model
        self.model = full_model
        student_trainer = self._train_model(callbacks)

        self.model.init_metrics()

        student_trainer.test(model=self.model, datamodule=self.datamodule.dataset)
        self.logger.info("Student model tested!")

        student_acc = self.model.test_acc.compute().cpu().numpy().tolist()
        train_results["student_accuracy"] = student_acc
        self.logger.info(f"Student accuracy: {student_acc}")

        self.logger.info(f"Stage {self.status.current_stage + 1}/{self.params.num_train_stages} "
                         f"|| Labeled set size {self.datamodule.labeled_pool_size} done: "
                         f"Student test acc {student_acc}, teacher test acc: {teacher_acc}")
        return train_results

    def _train_model(self, callbacks):
        model_trainer: Trainer = self.params.optimization_definition_set.trainer_definition.instantiate(
            logger=self.phase_logger,
            callbacks=callbacks,
            resume_from_checkpoint=self._resume_from_checkpoint)
        model_trainer.fit(model=self.model, datamodule=self.datamodule.dataset)
        if model_trainer.state == TrainerState.INTERRUPTED:
            raise KeyboardInterrupt

        self._resume_from_checkpoint = None

        self.logger.info("Model fit done!")
        return model_trainer

    def _test_model(self, callbacks):
        results = {}
        model_trainer: Trainer = self.params.optimization_definition_set.trainer_definition.instantiate(
            logger=self.phase_logger,
            callbacks=callbacks,
            resume_from_checkpoint=self._resume_from_checkpoint)

        self.model.init_metrics()
        handle, layer_to_inspect = self._register_attention_inspection()

        model_trainer.test(model=self.model, datamodule=self.datamodule.dataset)
        self.logger.info("Extended model tested!")

        acc = self.model.test_acc.compute().cpu().numpy().tolist()
        acc_5 = self.model.test_5_acc.compute().cpu().numpy().tolist()
        results["fr_accuracy"] = acc
        results["fr_accuracy_top5"] = acc_5

        results.update({f"fr_{key}": value for key, value in model_trainer.logged_metrics.items() if TrainStage.Test.value in key})

        self.logger.info(f"Stage {self.status.current_stage + 1}/{self.params.num_train_stages} "
                         f"|| Labeled set size {self.datamodule.labeled_pool_size} done: "
                         f"Extended model test acc {acc}")

        if self.params.robustness_benchmark:
            self.logger.info("Running robustness benchmark!")
            robustness_dataloaders = self.datamodule.dataset.robustness_dataloader()
            for corruption_type, courrupted_dataloaders in robustness_dataloaders.items():
                for courruption_severity, severity_dataloader in courrupted_dataloaders.items():
                    self.model.init_metrics()
                    model_trainer.test(model=self.model, test_dataloaders=[severity_dataloader])
                    acc = self.model.test_acc.compute()
                    results[f"fr_accuracy_{corruption_type}_{courruption_severity}"] = acc.cpu().numpy().tolist()
                    self.logger.info(f"{corruption_type} - {courruption_severity}: {acc}")

        return results

    def _train_iid(self, callbacks):
        iid_model = self.iid_model
        self.logger.info("Training the IID model!")
        iid_trainer: Trainer = self.params.optimization_definition_set.trainer_definition.instantiate(
            logger=self.phase_logger,
            callbacks=callbacks,
            resume_from_checkpoint=self._resume_from_checkpoint)
        iid_trainer.fit(model=iid_model, datamodule=self.datamodule.dataset)
        if iid_trainer.state == TrainerState.INTERRUPTED:
            raise KeyboardInterrupt

        self._resume_from_checkpoint = None

        self.logger.info("IID model fit done!")
        return iid_trainer

    def _test_iid(self, callbacks):
        results = {}
        iid_model = self.iid_model
        if iid_model is not None:
            self.logger.info("Testing the IID part!")
            iid_model.init_metrics()
            iid_trainer: Trainer = self.params.optimization_definition_set.trainer_definition.instantiate(
                logger=self.phase_logger,
                callbacks=callbacks,
                resume_from_checkpoint=self._resume_from_checkpoint)
            iid_trainer.test(model=iid_model, datamodule=self.datamodule.dataset)

            self.logger.info("IID model tested!")
            acc = iid_model.test_acc.compute()

            results["iid_accuracy"] = acc.cpu().numpy().tolist()
            self.logger.info(
                f"Stage {self.status.current_stage + 1}/{self.params.num_train_stages} "
                f"|| Label set size {self.datamodule.labeled_pool_size} done: "
                f"IID model test acc {acc}")

            if self.params.robustness_benchmark:
                self.logger.info("Running robustness benchmark!")
                robustness_dataloaders = self.datamodule.dataset.robustness_dataloader()
                for corruption_type, courrupted_dataloaders in robustness_dataloaders.items():
                    for courruption_severity, severity_dataloader in courrupted_dataloaders.items():
                        iid_model.init_metrics()
                        iid_trainer.test(model=iid_model, test_dataloaders=[severity_dataloader])
                        acc = iid_model.test_acc.compute()
                        results[f"iid_accuracy_{corruption_type}_{courruption_severity}"] = acc.cpu().numpy().tolist()
                        self.logger.info(f"{corruption_type} - {courruption_severity}: {acc}")
        else:
            self.logger.info("Can't find the IID part, unable to test it!")
        return results

    def _train_iid_head(self, callbacks):
        iid_model = self.iid_model
        self.logger.info("Training the IID head!")
        iid_model.backbone.freeze()
        head_trainer: Trainer = self.params.optimization_definition_set.trainer_definition.instantiate(
            logger=self.phase_logger,
            callbacks=callbacks,
            resume_from_checkpoint=self._resume_from_checkpoint)
        head_trainer.fit(model=iid_model, datamodule=self.datamodule.dataset)
        iid_model.backbone.unfreeze()
        if head_trainer.state == TrainerState.INTERRUPTED:
            raise KeyboardInterrupt

        self._resume_from_checkpoint = None

        self.logger.info("IID model fit done!")
        return head_trainer

    def _register_attention_inspection(self):
        try:
            self.model.clean_inspected_layers()
            layer_to_inspect = "model.mpn_net.model.mpn_layers.layer_0.trans.alpha_dropout"
            handle = self.model.inspect_layer_output(layer_to_inspect)
            self.logger.info("Attention inspection is registered!")
            return handle, layer_to_inspect
        except AttributeError:
            return None, None

    def query(self):
        # if not self.params.optimization_definition_set.trainer_definition.hyperparams.fast_dev_run:
        query_strategy = self.params.query_definition.instantiate(self._current_stage_folder_path)
        query_strategy.query(self.query_model,
                             self.datamodule,
                             self.params.num_new_labeled_samples_per_stage,
                             self.params.num_samples_to_evaluate_per_stage,
                             self.params.query_ascending)
        # else:
        #     self.datamodule.label_samples(self.params.num_new_labeled_samples_per_stage)

        self.datamodule.status.dumps_to_file(
            self._get_checkpoint(self.status.current_stage + 1, self.DATA_CHECKPOINT_NAME))

    def clean_up(self):
        if not self.params.keep_checkpoints:
            checkpoints = glob.glob(os.path.join(self.training_folder_path, '*', self.MODEL_CHECKPOINT_NAME))
            for filePath in checkpoints:
                try:
                    self.logger.info(f"Removing checkpoint {filePath}")
                    os.remove(filePath)
                except IOError as e:
                    self.logger.error(f"Error while deleting file {filePath}, {e}")

    def _log_command(self):
        self.logger.info(f"Starting command: {' '.join(sys.argv)}")

    def add_stage_results(self, stage_results: dict):
        stage_results = pd.DataFrame(stage_results, index=[0])
        self._training_results = self.training_results.merge(stage_results,
                                                             on=self.training_results.columns.tolist(),
                                                             how="outer")

    def log_results(self):
        """Logs the results to file"""
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        self.training_results.to_csv(self.training_result_file, index=False)

    def plot_results(self):
        """Plots the results to file"""
        self.logger.info("Plotting the results")
        viz_path = os.path.join(self.training_folder_path, "viz")
        os.makedirs(viz_path, exist_ok=True)

        training_results_cleared_of_junk = {stage_result["size_of_labelled_pool"]: stage_result["accuracy"]
                                            for stage_result in
                                            self.training_results.values()}

        fig = px.line(x=list(training_results_cleared_of_junk.keys()),
                      y=list(training_results_cleared_of_junk.values()),
                      labels={'x': 'Number of images', 'y': 'Test accuracy'},
                      title='Active learning')
        fig.write_html(os.path.join(viz_path, "active_learning.html"))

    def get_definition(self):
        return ActiveTrainerDefinition(self.params, self.status)

    def _get_status(self, status: Optional[ActiveTrainerStatus] = None) -> ActiveTrainerStatus:
        if status is None:
            return ActiveTrainerStatus()
        else:
            return status

    @property
    def is_benchmark(self):
        return self.params.run_id is not None

    @property
    def training_name(self):
        if self._training_name is None:
            if self.status.training_folder_path is None:
                if self.is_benchmark:
                    self._training_name = f"run_{self.params.run_id}"
                else:
                    if self.params.description is None:
                        self._training_name = f"training_{self.status.job_id}_{get_timestamp()}"
                    else:
                        self._training_name = f"training_{self.params.description}_{self.status.job_id}_{get_timestamp()}"
            elif self.params.logging_root in self.status.training_folder_path:
                self._training_name = os.path.basename(self.status.training_folder_path)
            else:
                raise ValueError(f"The defined training folder path ({self.status.training_folder_path}) "
                                 f"is not in the defined root folder ({self.params.logging_root})!")
        return self._training_name

    @property
    def training_scope(self):
        if self._training_scope is None:
            if self.status.training_folder_path is None:
                scopes = [self.root_scope,
                          self.params.optimization_definition_set.data_definition.type.value,
                          self.params.optimization_definition_set.model_definition.type.value]
                if self.is_benchmark:
                    if self.params.description is None:
                        scopes.append(f"benchmark_{self.params.benchmark_id}")
                    else:
                        scopes.append(f"benchmark_{self.params.description}_{self.params.benchmark_id}")

                self._training_scope = os.path.join(*scopes)
            elif self.params.logging_root in self.status.training_folder_path:
                self._training_scope = os.path.dirname(
                    os.path.relpath(self.status.training_folder_path, self.params.logging_root))
            else:
                raise ValueError(f"The defined training folder path ({self.status.training_folder_path}) "
                                 f"is not in the defined root folder ({self.params.logging_root})!")
        return self._training_scope

    @property
    def root_scope(self):
        if self.params.scope is None:
            return self.DEFAULT_SCOPE
        else:
            return self.params.scope

    @property
    def benchmark_folder_path(self):
        if self._benchmark_folder_path is None:
            if self.is_benchmark:
                self._benchmark_folder_path = os.path.join(self.params.logging_root, self.training_scope)
            else:
                self._benchmark_folder_path = self.training_folder_path
        return self._benchmark_folder_path

    @property
    def training_folder_path(self):
        if self.status.training_folder_path is None:
            self.status.training_folder_path = os.path.join(self.params.logging_root, self.training_scope,
                                                            self.training_name)
            os.makedirs(self.status.training_folder_path, exist_ok=True)
        return self.status.training_folder_path

    @property
    def training_result_file(self):
        if self._training_result_file is None:
            if self.is_benchmark:
                self._training_result_file = os.path.join(self.benchmark_folder_path,
                                                          f"{self.RESULTS_FILE}_{self.params.run_id}.{self.RESULTS_EXT}")
            else:
                self._training_result_file = os.path.join(self.benchmark_folder_path,
                                                          f"{self.RESULTS_FILE}.{self.RESULTS_EXT}")
        return self._training_result_file

    @property
    def training_log_file(self):
        if self._training_log_file is None:
            self._training_log_file = os.path.join(self.training_folder_path, self.LOG_FILE)
        return self._training_log_file

    @property
    def training_params_file(self):
        if self._training_params_file is None:
            self._training_params_file = os.path.join(self.benchmark_folder_path, self.PARAMS_FILE)
        return self._training_params_file

    def setup_logger(self):
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
        file_handler = logging.FileHandler(self.training_log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    @property
    def training_results(self):
        if self._training_results is None:
            if os.path.exists(self.training_result_file):
                self._training_results = pd.read_csv(self.training_result_file)
            else:
                self._training_results = pd.DataFrame(columns=['stage_id', 'size_of_labeled_pool'])
        return self._training_results

    @property
    def iid_model(self):
        try:
            return self.model.model.iid_net
        except AttributeError as e:
            self.logger.warning(f"Unable to use IID model during inference{e}!")
            return None

    @property
    def query_model(self):
        if self._trainer_type == TrainerType.LearningLossTrainer:
            if self.params.query_definition.type == QueryType.ModelQuery:
                return self.model
            else:
                return self.model.model.main_model
        else:
            if self.params.iid_inference:
                iid_model = self.iid_model
                if iid_model is not None:
                    return iid_model
        return self.model


    @property
    def inference_model(self):
        if self.params.iid_inference:
            iid_model = self.iid_model
            if iid_model is not None:
                return iid_model
        return self.model

    def _get_checkpoint(self, stage: int, checkpoint_name: str):
        return os.path.join(self.training_folder_path, self.status.get_stage_id(stage), checkpoint_name)

    def _get_existing_checkpoint(self, stage: int, checkpoint_name: str):
        checkpoint = self._get_checkpoint(stage, checkpoint_name)
        if os.path.exists(checkpoint):
            return checkpoint
        return None

    def _initialize_model(self):
        return self.params.optimization_definition_set.model_definition.instantiate()

    def _initialize_datamodule(self):
        if self.params.initial_pool_file is not None:
            datamodule_status = ActiveDataModuleStatus.loads_from_file(self.params.initial_pool_file)
        else:
            datamodule_status = ActiveDataModuleStatus(self.status.status_file, self.status.job_id)
        datamodule_params = self.params.datamodule_hyperparams

        datamodule_definition = ActiveDataModuleDefinition(datamodule_params, datamodule_status)

        return datamodule_definition.instantiate(self.params.optimization_definition_set.data_definition)

    def _restore_model(self, stage: int):
        model = self._initialize_model()

        model_checkpoint = self._get_checkpoint(stage, self.MODEL_CHECKPOINT_NAME)
        if os.path.exists(model_checkpoint):
            model = model.load_from_checkpoint(model_checkpoint)
        return model

    def _restore_datamodule(self, stage: int):
        datamodule_status = ActiveDataModuleStatus.loads_from_file(
            self._get_checkpoint(stage, self.DATA_CHECKPOINT_NAME))
        datamodule_params = self.params.datamodule_hyperparams

        datamodule_definition = ActiveDataModuleDefinition(datamodule_params, datamodule_status)

        datamodule = datamodule_definition.instantiate(self.params.optimization_definition_set.data_definition)
        datamodule.update_train_sampler(self.model.feature_model, self.params.optimization_definition_set.trainer_definition.hyperparams.device)
        datamodule.update_test_sampler(self.model.feature_model, self.params.optimization_definition_set.trainer_definition.hyperparams.device)
        return datamodule

    def _create_stage_logger(self, stage: int):
        return ClassificationLogger(save_dir=self.params.logging_root,
                                    name=os.path.join(self.training_scope, self.training_name),
                                    version=self.status.get_stage_id(stage))

    @property
    def _current_stage_folder_path(self):
        return os.path.join(self.training_folder_path, self.status.get_stage_id(self.status.current_stage))

    def _update_phase_logger(self):
        self.phase_logger.end_monitoring()
        self.phase_logger = self._create_stage_logger(self.status.current_stage)
        self.phase_logger.start_monitoring(self._phase_log_path)

    @property
    def _phase_log_path(self):
        return f"{self.status.state.value}.log"

    def _log_params(self):
        if not os.path.exists(self.training_params_file):
            self.params.dumps_to_file(self.training_params_file)

    @property
    def _attention_histogram_path(self):
        return os.path.join(self._current_stage_folder_path, self.ATTENTION_HISTOGRAM)

    def _log_attention_histogram(self, attention_histogram):
        attention_df = pd.DataFrame(attention_histogram.T)
        attention_df.to_csv(self._attention_histogram_path, index=False)

    @property
    def _most_influential_samples_path(self):
        return os.path.join(self._current_stage_folder_path, self.MOST_INFLUENTIAL_SAMPLES)

    def _log_most_influential_samples(self, most_influential_samples):
        most_influential_samples_df = pd.DataFrame(most_influential_samples)
        most_influential_samples_df.to_csv(self._most_influential_samples_path, index=False)

    def _set_seed(self):
        seed_everything(self.params.seed)


def active_learning():
    """
    Example usage
    """
    # -----------------------------------------------
    # Hyperparameters
    # -----------------------------------------------
    optimization_definition_set = OptimizationDefinitionSet(
        data_definition=CIFAR10Definition(CIFAR10HyperParameterSet(val_ratio=0.0)),
        model_definition=DummyModuleDefinition(DummyModuleHyperParameterSet(
            optimizer_definition=SGDOptimizerDefinition(
                SGDOptimizerHyperParameterSet(
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=5e-4)),
            scheduler_definition=MultiStepLRSchedulerDefinition(
                MultiStepLRSchedulerHyperParameterSet(
                    milestone_ratios=[160]
                )
            ))),
        trainer_definition=PL_TrainerDefinition(
            PL_TrainerHyperParameterSet(max_epochs=5, fast_dev_run=True)),
        seed=1234)
    optimization_definition_set.data_definition.hyperparams.batch_size = 100
    # -----------------------------------------------
    # Callbacks
    # -----------------------------------------------
    callbacks = None

    active_trainer_params = ActiveTrainerHyperParameterSet(
        num_train_stages=2,
        inference_labeled_ratio=0.9,
        return_uncertainties=True,
        return_logits=False,
        benchmark_id=None,
        optimization_definition_set=optimization_definition_set)

    active_trainer = ActiveTrainer(active_trainer_params)
    result = active_trainer.train(callbacks)
    print(result)


if __name__ == '__main__':
    active_learning()
