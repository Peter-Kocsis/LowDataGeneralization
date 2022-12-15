import copy
import os
from argparse import ArgumentParser, Namespace
from typing import Optional

from lowdataregime.active_learning.active_datamodule import ActiveDataModuleHyperParameterSet
from lowdataregime.active_learning.active_trainer import ActiveTrainerHyperParameterSet, \
    ActiveTrainerDefinition, ActiveTrainerStatus
from lowdataregime.active_learning.query.coreset_query import CoreSetQueryDefinition, CoreSetQueryHyperParameterSet
from lowdataregime.active_learning.query.model_query import ModelQueryDefinition, ModelQueryHyperParameterSet
from lowdataregime.active_learning.query.multifactor_query import MultiFactorQueryDefinition, MultiFactorQueryHyperParameterSet
from lowdataregime.active_learning.query.multistage_query import MultiStageQueryHyperParameterSet, MultiStageQueryDefinition
from lowdataregime.active_learning.query.queries import QueryType
from lowdataregime.active_learning.query.random_query import RandomQueryDefinition
from lowdataregime.active_learning.query.uncertainty_query import UncertaintyQueryDefinition, \
    UncertaintyQueryHyperParameterSet
from lowdataregime.classification.data.caltech import Caltech101Definition, Caltech101HyperParameterSet, Caltech256Definition, \
    Caltech256HyperParameterSet
from lowdataregime.classification.data.cifar import CIFAR10Definition, CIFAR10HyperParameterSet, \
    CIFAR100Definition, CIFAR100HyperParameterSet
from lowdataregime.classification.model.convolutional.densenet import DenseNetDefinition, DenseNetHyperParameterSet
from lowdataregime.classification.model.convolutional.dml import DMLNetDefinition, DMLNetHyperParameterSet
from lowdataregime.classification.model.convolutional.efficientnet import EfficientNetB3Definition, \
    EfficientNetB3HyperParameterSet
from lowdataregime.classification.model.convolutional.kd import KDNetDefinition, KDNetHyperParameterSet
from lowdataregime.classification.model.convolutional.kdcl import KDCLNetDefinition, KDCLNetHyperParameterSet
from lowdataregime.classification.model.convolutional.pretrained_densenet import PretrainedDenseNetDefinition, \
    PretrainedDenseNetHyperParameterSet
from lowdataregime.classification.model.convolutional.vgg import VGG11Definition, VGG11HyperParameterSet
from lowdataregime.classification.model.convolutional.wide_resnet import WideResNetDefinition, WideResNetHyperParameterSet
from lowdataregime.classification.model.feature_refiner.feature_refiner import FeatureRefinerDefinition, \
    FeatureRefinerHyperParameterSet
from lowdataregime.classification.model.feature_refiner.fr_net import FRNetHyperParameterSet, FRNetDefinition
from lowdataregime.classification.model.feature_refiner.simple_fr_net import SimpleFRNetDefinition, \
    SimpleFRNetHyperParameterSet
from lowdataregime.classification.model.learning_loss.learning_loss import LearningLossDefinition, \
    LearningLossHyperParameterSet
from lowdataregime.classification.model.learning_loss.lossnet import LossNetDefinition, LossNetHyperParameterSet
from lowdataregime.classification.sampling.class_based_sampler import ClassBasedSamplerDefinition, \
    ClassBasedSamplerHyperParameterSet
from lowdataregime.classification.sampling.combine_sampler import CombineSamplerDefinition, CombineSamplerHyperParameterSet
from lowdataregime.classification.data.datas import DataModuleType
from lowdataregime.classification.sampling.region_sampler import RegionSamplerDefinition, RegionSamplerHyperParameterSet
from lowdataregime.classification.sampling.sampler import SamplerType, SubsetSequentialSamplerDefinition, \
    SubsetRandomSamplerDefinition
from lowdataregime.classification.data.transform import ComposeTransformDefinition, \
    ComposeTransformHyperParameterSet, RandomHorizontalFlipTransformDefinition, \
    RandomCropTransformDefinition, ToTensorTransformDefinition, NormalizeTransformDefinition, \
    NormalizeHyperParameterSet, RandomCropTransformHyperParameterSet, ResizeTransformDefinition, \
    ResizeHyperParameterSet, RepeatTransformDefinition, RepeatHyperParameterSet
from lowdataregime.classification.loss.batch_center_loss import BatchCenterLossDefinition, BatchCenterLossHyperParameterSet
from lowdataregime.classification.loss.center_loss import CenterLossDefinition, CenterLossHyperParameterSet
from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.loss.losses import CrossEntropyDefinition, DMLLossDefinition, KDCLLossDefinition, \
    KDLossDefinition
from lowdataregime.classification.model.convolutional.pretrained_resnet import PretrainedResNetDefinition, \
    PretrainedResNetHyperParameterSet, ResNetType
from lowdataregime.classification.model.convolutional.resnet import ResNet18Definition, \
    ResNet18HyperParameterSet, ResNet34Definition, ResNet34HyperParameterSet, ResNet50HyperParameterSet, ResNet50Definition
from lowdataregime.classification.model.dummy.fc_classifier import DummyModuleDefinition, \
    DummyModuleHyperParameterSet
from lowdataregime.classification.model.models import ModelType, PretrainingType
from lowdataregime.classification.optimizer.optimizers import SGDOptimizerDefinition, \
    SGDOptimizerHyperParameterSet, AdamOptimizerDefinition, OptimizerType, AdamOptimizerHyperParameterSet, \
    RAdamOptimizerDefinition, RAdamOptimizerHyperParameterSet
from lowdataregime.classification.optimizer.schedulers import MultiStepLRSchedulerDefinition, \
    MultiStepLRSchedulerHyperParameterSet
from lowdataregime.classification.trainer.fixmatch_trainer import FixMatchTrainerDefinition, FixMatchTrainerHyperParameterSet
from lowdataregime.classification.trainer.learning_loss_trainer import LearningLossTrainerDefinition, \
    LearningLossTrainerHyperParameterSet
from lowdataregime.classification.trainer.trainers import PL_TrainerDefinition, \
    PL_TrainerHyperParameterSet, DeviceType, TrainerType
from lowdataregime.experiment.experiment import Experiment, ExperimentInfo
from lowdataregime.parameters.params import OptimizationDefinitionSet
from lowdataregime.utils.distance import DistanceType, EuclideanDistanceDefinitionSet, CosineDistanceDefinitionSet, \
    CorrelationDistanceDefinitionSet
from lowdataregime.utils.utils import str2bool, remove_prefix, str2int


class ActiveLearningExperimentInfo(ExperimentInfo, ActiveTrainerDefinition):
    @classmethod
    def from_str(cls, info: str):
        return cls.loads(info)

    def to_str(self) -> str:
        return self.dumps()

    @classmethod
    def from_definition(cls, param):
        return cls(param.hyperparams, param.status)


class ActiveLearningExperiment(Experiment):

    @classmethod
    def add_argparse_args(cls, parent_parser=ArgumentParser(add_help=False)):
        """
        Defines the arguments of the class
        :param parent_parser: The parser which should be extended
        :return: The extended parser
        """
        super_parser = super(cls, cls).add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser],
                                description="Script for aquiring calibration images from video.",
                                add_help=True
                                )
        # --------------------------- ACTIVE LEARNING ---------------------------
        parser.add_argument(
            "-rid", "--run_id", type=int, default=None,
            help="Run ID if running benchmark"
        )
        parser.add_argument(
            "-desc", "--description", type=str, default=None,
            help="Description of the training"
        )
        parser.add_argument(
            "-rs", "--root_scope", type=str, default=None,
            help="Scope of the experiment"
        )
        parser.add_argument(
            "-kc", "--keep_checkpoints", action="store_true", default=False,
            help="Indicates whether to keep the checkpoints or not"
        )
        parser.add_argument(
            "-rb", "--robustness_benchmark", action="store_true", default=False,
            help="Indicates whether to run robustness benchmarking or not"
        )

        # -------------------------------- MODEL --------------------------------
        parser.add_argument(
            "-m", "--model", type=ModelType, choices=list(ModelType), required=True,
            help="Model to use in the active learning"
        )
        parser.add_argument(
            "-bbm", "--backbone_model", type=ModelType, choices=list(ModelType), default=ModelType.ResNet18,
            help="Model to use as GeneralNet backbone"
        )
        parser.add_argument(
            "-op", "--optimized_parameters", action="store_true", default=False,
            help="Indicates whether to use the best hyperparameters automatically or not"
        )
        parser.add_argument(
            "-hb", "--head_bias", type=str2bool, default=True,
            help="Using the head bias or not"
        )
        parser.add_argument(
            "-conc", "--concat", action="store_true", default=False,
            help="Concatenate the heads in multi-headed attention"
        )
        parser.add_argument(
            "-sd", "--skip_downsampling", action="store_true", default=False,
            help="Skip the downsampling in transformer layer, causing feature size expansion"
        )
        parser.add_argument(
            "-bet", "--beta", type=str2bool, default=False,
            help="Will scale the message passing and root weighting"
        )
        parser.add_argument(
            "-rw", "--root_weight", type=str2bool, default=True,
            help="Will scale the message passing and root weighting"
        )
        parser.add_argument(
            "-gm", "--gradient_multiplier", type=float, default=0.0,
            help="The multiplier of the GradientGate before the IID head"
        )
        parser.add_argument(
            "-th", "--train_head", action="store_true", default=False,
            help="Indicates whether to train the iid head or not"
        )
        parser.add_argument(
            "-pt", "--pretraining_type", type=PretrainingType, choices=list(PretrainingType),
            default=PretrainingType.NONE,
            help="Indicates whether to use pretrained ResNet18 or not"
        )
        parser.add_argument(
            "-rt", "--retrained", action="store_true", default=False,
            help="Indicates whether to use pretrained ResNet18 but not pretrained"
        )
        parser.add_argument(
            "-dg", "--distinct_graph", action="store_true", default=False,
            help="Indicates whether to use distinct graph or not"
        )
        parser.add_argument(
            "-ean", "--edge_attrib_normalize", action="store_true", default=False,
            help="Indicates whether to normalize the edge attributes or not"
        )
        parser.add_argument(
            "-ed", "--edge_dimension", type=int, default=None,
            help="Defines the edge dimension in MPN"
        )
        parser.add_argument(
            "-qkbs", "--query_key_bias", type=str2bool, default=True,
            help="Whether to use bias in query, key, skip layers in attention"
        )
        parser.add_argument(
            "-il", "--intermediate_layers", type=str, nargs='+', default=[],
            help="The loss definitions to use"
        )
        parser.add_argument("-usrwo", "--ultimate_super_root_weight_off", type=str2bool, default=False,
                            help="Whether to shut down the skip connection in the forward of trans conv layer"
                            )
        parser.add_argument("-dmp", "--disable_mp", type=str2bool, default=False,
                            help="Whether to shut down the MP in the Transformer model"
                            )
        parser.add_argument("-dn1", "--disable_norm1", type=str2bool, default=False,
                            help="Whether to shut down the MP in the Transformer model"
                            )
        parser.add_argument("-dn2", "--disable_norm2", type=str2bool, default=False,
                            help="Whether to shut down the MP in the Transformer model"
                            )
        parser.add_argument("-dl1", "--disable_layer1", type=str2bool, default=False,
                            help="Whether to shut down the MP in the Transformer model"
                            )
        parser.add_argument("-dl2", "--disable_layer2", type=str2bool, default=False,
                            help="Whether to shut down the MP in the Transformer model"
                            )
        parser.add_argument(
            "-nlc", "--num_last_channels", type=int, default=512,
            help=""
        )
        parser.add_argument(
            "-nc", "--num_channels", type=int, nargs='+', default=[64, 128, 256, 512],
            help=""
        )
        parser.add_argument(
            "-nil", "--num_inner_layers", type=int, default=1,
            help=""
        )

        # -------------------------------- DATA --------------------------------
        parser.add_argument(
            "-d", "--data_type", type=DataModuleType, choices=list(DataModuleType), default=DataModuleType.CIFAR10,
            help="Dataset to use in the active learning"
        )
        parser.add_argument(
            "-ndw", "--num_data_workers", type=int, default=4,
            help="Number of unlabelled samples to label in each stage"
        )
        parser.add_argument(
            "-df", "--duplication_factor", type=int, default=1, help="Factor of dataset duplication"
        )
        parser.add_argument(
            "-ds", "--data_seed", type=int, default=0, help="Data seed"
        )
        parser.add_argument(
            "-tspc", "--train_samples_per_class", type=int, default=None,
            help="The number of training samples in each class"
        )

        # -------------------------------- TRAIN --------------------------------
        parser.add_argument(
            "-tt", "--trainer_type", type=TrainerType, choices=list(TrainerType), default=TrainerType.PL_Trainer,
            help="Defines the type of the trainer"
        )
        parser.add_argument(
            "-nts", "--num_train_stages", type=int, default=10, help="Number of active learning stages"
        )
        parser.add_argument(
            "-nis", "--num_initial_samples", type=int, default=1000,
            help="Number of samples in the initial labeled pool"
        )
        parser.add_argument(
            "-ipf", "--initial_pool_file", type=str, default=None,
            help="Number of samples in the initial labeled pool"
        )
        parser.add_argument(
            "-nnlsps", "--num_new_labeled_samples_per_stage", type=int, default=1000,
            help="Number of unlabelled samples to label in each stage"
        )
        parser.add_argument(
            "-nsteps", "--num_samples_to_evaluate_per_training", type=int, default=None,
            help="Number of unlabelled samples to evaluate during labelling"
        )
        parser.add_argument(
            "-tfc", "--train_from_scratch", action="store_true", default=False,
            help="Indicates whether to train from scratch or not"
        )
        parser.add_argument(
            "-tgn", "--track_grad_norm", action="store_true", default=False,
            help="Indicates whether to track grad norm or not"
        )
        parser.add_argument(
            "-dev", "--device", type=DeviceType, choices=list(DeviceType), default=DeviceType.GPU,
            help="Defines the maximum number of epochs"
        )
        parser.add_argument(
            "-fdr", "--fast_dev_run", action="store_true", default=False,
            help="Indicates whether to run fast dev run or not"
        )
        parser.add_argument(
            "-trast", "--train_sampler_type", type=SamplerType, choices=list(SamplerType),
            default=SamplerType.SubsetRandomSampler,
            help="Defines the type of the sampler used during training"
        )
        parser.add_argument(
            "-tesst", "--test_sampler_type", type=SamplerType, choices=list(SamplerType),
            default=SamplerType.SubsetRandomSampler,
            help="Defines the type of the sampler used during testing"
        )
        parser.add_argument(
            "-unlst", "--unlabeled_sampler_type", type=SamplerType, choices=list(SamplerType),
            default=SamplerType.SubsetSequentialSampler,
            help="Defines the type of the sampler used during query"
        )
        parser.add_argument(
            "-tsnc", "--train_sampler_num_classes", type=int, default=None, help="Number of classes in batch"
        )
        parser.add_argument(
            "-ot", "--optimizer_type", type=OptimizerType, choices=list(OptimizerType), default=OptimizerType.SGD,
            help="Query strategy used in the active learning"
        )
        parser.add_argument(
            "-ld", "--loss_definition", nargs='+', default=None,
            help="The loss definitions to use"
        )
        parser.add_argument(
            "-lw", "--loss_weights", type=float, nargs='+', default=None,
            help="The loss weights to use"
        )
        parser.add_argument(
            "-bclm", "--batch_center_loss_metric", type=DistanceType, choices=list(DistanceType),
            default=DistanceType.EUCLIDEAN_DISTANCE,
            help="The metric used by batch center loss"
        )
        parser.add_argument(
            "-s", "--seed", type=str2int, default=0, help="Defines the seed"
        )
        parser.add_argument(
            "-ve", "--validate_epochs", type=int, default=None,
            help="The number of epochs after doing validation"
        )

        # -------------------------------- QUERY --------------------------------
        parser.add_argument(
            "-qt", "--query_type", type=QueryType, choices=list(QueryType), default=QueryType.UncertaintyQuery,
            help="Query strategy used in the active learning"
        )
        parser.add_argument(
            "-qf", "--query_factors", type=QueryType, nargs='+', choices=list(QueryType), default=None,
            help="Query factors of multifactor query"
        )
        parser.add_argument(
            "-qs", "--query_stages", type=int, nargs='+', default=None,
            help="Number of samples in each query stages"
        )
        parser.add_argument(
            "-cqm", "--coreset_query_metric", type=DistanceType, choices=list(DistanceType),
            default=DistanceType.EUCLIDEAN_DISTANCE,
            help="Metric of coreset query"
        )
        parser.add_argument(
            "-itsr", "--inference_training_sample_ratio", type=float, default=0.0,
            help="Ratio of the labelled and unlabelled samples during inference of unlabelled samples"
        )
        parser.add_argument(
            "-t", "--temperature", type=float, default=1.0, help="Temperature of calibration"
        )
        parser.add_argument(
            "-iidi", "--iid_inference", action="store_true", default=False,
            help="Indicates whether to use the IID backbone for inference or not"
        )
        parser.add_argument(
            "-iidt", "--iid_training", action="store_true", default=False,
            help="Indicates whether to train only the iid part or not"
        )
        parser.add_argument(
            "-lq", "--log_query", action="store_true", default=False,
            help="Indicates whether to log query info or not"
        )
        parser.add_argument(
            "-noe", "--num_of_estimations", type=int, default=10,
            help="The number of Monte Carlo estimates of attention query"
        )
        parser.add_argument(
            "-ascnd", "--query_ascending", type=str2bool, default=False,
            help="The query metric will be evaluated in ascending order"
        )

        # -------------------------------- PARAMS --------------------------------
        parser.add_argument(
            "-bs", "--batch_size", type=int, default=None, help="Batch size"
        )

        parser.add_argument(
            "-fs", "--feature_size", type=int, default=None, help="Feature size of MPN"
        )
        parser.add_argument(
            "-nmp", "--num_message_passings", type=int, default=None,
            help="Number of message passings by MPN"
        )
        parser.add_argument(
            "-nh", "--num_heads", type=int, default=None,
            help="Number of heads of attention during message passings"
        )
        parser.add_argument(
            "-ast", "--attention_scaling_threshold", type=float, default=None,
            help="Defines the scale of attention scaling"
        )
        parser.add_argument(
            "-me", "--max_epochs", type=int, default=None, help="Defines the maximum number of epochs"
        )

        parser.add_argument(
            "-lrate", "--learning_rate", type=float, default=None,
            help="The learning rate"
        )
        parser.add_argument(
            "-lratemsr", "--lr_milestone_ratio", type=float, nargs='+', default=None,
            help="The milestone ratio for learning rate scheduling"
        )
        parser.add_argument(
            "-lrateg", "--lr_milestone_gamma", type=float, default=None,
            help="The learning rate decay"
        )
        parser.add_argument(
            "-mom", "--momentum", type=float, default=None,
            help="The momentum"
        )
        parser.add_argument(
            "-wd", "--weight_decay", type=float, default=None,
            help="The weight decay"
        )

        return parser

    def prepare(self, experiment_info_str: Optional[str]):
        experiment_info = self._get_active_trainer_definition(experiment_info_str)
        self.active_trainer = experiment_info.instantiate(experiment=self)

    def experiment_info(self):
        return ActiveLearningExperimentInfo.from_definition(self.active_trainer.get_definition())

    def execute(self):
        print(f"Running training: {self.active_trainer.params.dumps()}")
        result = self.active_trainer.train()
        print(result)

    def _get_active_trainer_definition(self, experiment_info_str: Optional[str]):
        if experiment_info_str is None:
            active_trainer_params = self._get_active_trainer_params()
            active_trainer_status = ActiveTrainerStatus(self._experiment_status_path, self.job_id)
            experiment_info = ActiveLearningExperimentInfo(active_trainer_params, active_trainer_status)
        else:
            experiment_info = ActiveLearningExperimentInfo.from_str(experiment_info_str)
        return experiment_info

    def _get_active_trainer_params(self):
        optimization_definition_set = self._get_optimization_set()
        query_definition_set = self._get_query_definition_set()
        datamodule_hyperparams = self._get_datamodule_hyperparams()
        benchmark_params = self._get_benchmark_params()

        return ActiveTrainerHyperParameterSet(
            num_train_stages=self.arguments.num_train_stages,
            num_initial_samples=self.arguments.num_initial_samples,
            num_new_labeled_samples_per_stage=self.arguments.num_new_labeled_samples_per_stage,
            num_samples_to_evaluate_per_stage=self.arguments.num_samples_to_evaluate_per_training,
            train_from_scratch=self.arguments.train_from_scratch,
            optimization_definition_set=optimization_definition_set,
            datamodule_hyperparams=datamodule_hyperparams,
            query_definition_set=query_definition_set,
            logging_root=self.arguments.logging_root,
            iid_inference=self.arguments.iid_inference,
            iid_training=self.arguments.iid_training,
            benchmark_id=benchmark_params.benchmark_id,
            run_id=benchmark_params.run_id,
            description=self.arguments.description,
            scope=self.arguments.root_scope,
            train_head=self.arguments.train_head,
            keep_checkpoints=self.arguments.keep_checkpoints,
            initial_pool_file=self.arguments.initial_pool_file,
            query_ascedning=self.arguments.query_ascending,
            robustness_benchmark=self.arguments.robustness_benchmark
        )

    def _get_benchmark_params(self):
        benchmark_params = Namespace()
        benchmark_params.run_id = self.arguments.run_id
        if benchmark_params.run_id is None:
            benchmark_params.benchmark_id = None
        else:
            if "SLURM_ARRAY_JOB_ID" in os.environ:
                benchmark_params.benchmark_id = os.environ["SLURM_ARRAY_JOB_ID"]
            else:
                self.module_logger.debug(f"SLURM_ARRAY_JOB_ID not found in {os.environ}")
                benchmark_params.benchmark_id = self.job_id or -1
                self.module_logger.warning(f"Benchmark ID not found, using default value: {benchmark_params.benchmark_id}")

        return benchmark_params

    def _get_optimization_set(self):
            self._update_default_params()
            data_definition = self._get_data_definition()
            model_definition = self._get_model_definition(data_definition.hyperparams.num_classes)
            trainer_definition = self._get_trainer_definition()
            return OptimizationDefinitionSet(
                data_definition=data_definition,
                model_definition=model_definition,
                trainer_definition=trainer_definition,
                seed=self.arguments.seed)

    def _get_query_definition_set(self):
        temperature = self.arguments.temperature

        if self.arguments.model == ModelType.GeneralNet:
            if self.arguments.iid_inference:
                layer_of_features = "model.backbone.flatten"
            else:
                layer_of_features = "model.mpn_net.model.mpn_layers.layer_0"
        elif self.arguments.model == ModelType.FeatureRefiner:
            if self.arguments.iid_inference:
                layer_of_features = "model.backbone.flatten"
            else:
                layer_of_features = "model.fr_net.model.fr"
        elif self.arguments.model == ModelType.ResNet18:
            layer_of_features = "model.backbone.flatten"
        else:
            layer_of_features = None

        query_definitions = \
            {
                QueryType.RandomQuery: RandomQueryDefinition(),
                QueryType.UncertaintyQuery: UncertaintyQueryDefinition(
                    UncertaintyQueryHyperParameterSet(
                        inference_training_sample_ratio=self.arguments.inference_training_sample_ratio,
                        temperature=temperature,
                        unlabeled_pool_sampler_definition=self._get_sampler_definition(
                            self.arguments.unlabeled_sampler_type),
                        log_query=self.arguments.log_query
                    )
                ),
                QueryType.CoreSetQuery: CoreSetQueryDefinition(
                    CoreSetQueryHyperParameterSet(
                        layer_of_features=layer_of_features,
                        distance_metric=self._get_distance_metric(self.arguments.coreset_query_metric)
                    )
                ),
                QueryType.ModelQuery: ModelQueryDefinition(
                    ModelQueryHyperParameterSet(
                        unlabeled_pool_sampler_definition=self._get_sampler_definition(
                            self.arguments.unlabeled_sampler_type),
                        log_query=self.arguments.log_query
                    )
                )
            }

        if self.arguments.query_factors is not None:
            query_factors = {query_factor.value: query_definitions[query_factor] for query_factor in
                             self.arguments.query_factors}
        else:
            query_factors = {}

        query_definitions[QueryType.MultiFactorQuery] = MultiFactorQueryDefinition(
            MultiFactorQueryHyperParameterSet(
                query_definitions=query_factors,
                log_query=self.arguments.log_query
            )
        )
        query_definitions[QueryType.MultiStageQuery] = MultiStageQueryDefinition(
            MultiStageQueryHyperParameterSet(
                query_definitions=query_factors,
                num_samples_of_query_stages=self.arguments.query_stages,
                log_query=self.arguments.log_query
            )
        )

        return query_definitions[self.arguments.query_type]

    def _get_datamodule_hyperparams(self):
        return ActiveDataModuleHyperParameterSet(
            initial_pool_seed=self.arguments.data_seed,
            train_sampler_definition=self._get_sampler_definition(self.arguments.train_sampler_type),
            test_sampler_definition=self._get_sampler_definition(self.arguments.test_sampler_type),
            use_validation=self.arguments.validate_epochs is not None)

    def _get_sampler_definition(self, sampler_type):
        if sampler_type in [SamplerType.CombineSampler, SamplerType.ClassBasedSampler]:
            assert self.arguments.train_sampler_num_classes is not None, "Number of classes in the batch is not defined"

            train_sampler_num_samples = self.arguments.batch_size // self.arguments.train_sampler_num_classes
            assert train_sampler_num_samples * self.arguments.train_sampler_num_classes == self.arguments.batch_size, \
                f"Batch size ({self.arguments.batch_size}) is not divisible " \
                f"by the number of classes in batch ({self.arguments.train_sampler_num_classes})"
        else:
            train_sampler_num_samples = None

        if self.arguments.train_sampler_num_classes is not None:
            region_sampler_region_size = self.arguments.batch_size // self.arguments.train_sampler_num_classes
            assert region_sampler_region_size * self.arguments.train_sampler_num_classes == self.arguments.batch_size, \
                f"Batch size ({self.arguments.batch_size}) is not divisible " \
                f"by the number of classes in batch ({self.arguments.train_sampler_num_classes})"
        else:
            region_sampler_region_size = self.arguments.batch_size

        train_sampler_definitions = \
            {
                SamplerType.SubsetSequentialSampler: SubsetSequentialSamplerDefinition(),
                SamplerType.SubsetRandomSampler: SubsetRandomSamplerDefinition(),
                SamplerType.CombineSampler: CombineSamplerDefinition(
                    CombineSamplerHyperParameterSet(
                        num_classes_in_batch=self.arguments.train_sampler_num_classes,
                        num_samples_per_class=train_sampler_num_samples
                    )
                ),
                SamplerType.ClassBasedSampler: ClassBasedSamplerDefinition(
                    ClassBasedSamplerHyperParameterSet(
                        num_classes_in_batch=self.arguments.train_sampler_num_classes,
                        num_samples_per_class=train_sampler_num_samples
                    )
                ),
                SamplerType.RegionSampler: RegionSamplerDefinition(
                    RegionSamplerHyperParameterSet(
                        region_size=region_sampler_region_size,
                        num_data_workers=self.arguments.num_data_workers
                    )
                )
            }
        return train_sampler_definitions[sampler_type]

    def _get_data_definition(self):
        data_definitions = \
            {
                DataModuleType.CIFAR10: CIFAR10Definition(
                    CIFAR10HyperParameterSet(
                        num_workers=self.arguments.num_data_workers,
                        batch_size=self.arguments.batch_size,
                        use_kdcl_sampling=self.arguments.model == ModelType.KDCLNet,
                        val_ratio=0.0,
                        duplication_factor=self.arguments.duplication_factor,
                        train_transforms_def=ComposeTransformDefinition(
                            ComposeTransformHyperParameterSet([
                                RandomHorizontalFlipTransformDefinition(),
                                RandomCropTransformDefinition(
                                    RandomCropTransformHyperParameterSet(size=32, padding=4)),
                                ToTensorTransformDefinition(),
                                NormalizeTransformDefinition(
                                    NormalizeHyperParameterSet(mean=[0.4914, 0.4822, 0.4465],
                                                               std=[0.2023, 0.1994, 0.2010]))
                            ])),
                        val_transforms_def=None,
                        test_transforms_def=ComposeTransformDefinition(
                            ComposeTransformHyperParameterSet([
                                ToTensorTransformDefinition(),
                                NormalizeTransformDefinition(
                                    NormalizeHyperParameterSet(mean=[0.4914, 0.4822, 0.4465],
                                                               std=[0.2023, 0.1994, 0.2010]))
                            ])))),
                DataModuleType.CIFAR100: CIFAR100Definition(
                    CIFAR100HyperParameterSet(
                        num_workers=self.arguments.num_data_workers,
                        batch_size=self.arguments.batch_size,
                        use_kdcl_sampling=self.arguments.model == ModelType.KDCLNet,
                        val_ratio=0.0,
                        duplication_factor=self.arguments.duplication_factor,
                        train_transforms_def=ComposeTransformDefinition(
                            ComposeTransformHyperParameterSet([
                                RandomHorizontalFlipTransformDefinition(),
                                RandomCropTransformDefinition(
                                    RandomCropTransformHyperParameterSet(size=32, padding=4)),
                                ToTensorTransformDefinition(),
                                NormalizeTransformDefinition(
                                    NormalizeHyperParameterSet(mean=[0.5071, 0.4867, 0.4408],
                                                               std=[0.2675, 0.2565, 0.2761]))
                            ])),
                        val_transforms_def=None,
                        test_transforms_def=ComposeTransformDefinition(
                            ComposeTransformHyperParameterSet([
                                ToTensorTransformDefinition(),
                                NormalizeTransformDefinition(
                                    NormalizeHyperParameterSet(mean=[0.5071, 0.4867, 0.4408],
                                                               std=[0.2675, 0.2565, 0.2761]))
                            ])))),
                DataModuleType.CALTECH101: Caltech101Definition(
                    Caltech101HyperParameterSet(
                        num_workers=self.arguments.num_data_workers,
                        batch_size=self.arguments.batch_size,
                        val_ratio=0.0,
                        train_samples_per_class=self.arguments.train_samples_per_class,
                        train_transforms_def=ComposeTransformDefinition(
                            ComposeTransformHyperParameterSet([
                                ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                                RandomHorizontalFlipTransformDefinition(),
                                RandomCropTransformDefinition(
                                    RandomCropTransformHyperParameterSet(size=224, padding=16)),
                                ToTensorTransformDefinition(),
                                RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                                NormalizeTransformDefinition(
                                    NormalizeHyperParameterSet(mean=[0.5013, 0.4772, 0.4475],
                                                               std=[0.3331, 0.3277, 0.3343]))
                            ])),
                        val_transforms_def=None,
                        test_transforms_def=ComposeTransformDefinition(
                            ComposeTransformHyperParameterSet([
                                ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                                ToTensorTransformDefinition(),
                                RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                                NormalizeTransformDefinition(
                                    NormalizeHyperParameterSet(mean=[0.5013, 0.4772, 0.4475],
                                                               std=[0.3331, 0.3277, 0.3343]))
                            ])))),
                DataModuleType.CALTECH256: Caltech256Definition(
                    Caltech256HyperParameterSet(
                        num_workers=self.arguments.num_data_workers,
                        batch_size=self.arguments.batch_size,
                        val_ratio=0.0,
                        train_samples_per_class=self.arguments.train_samples_per_class,
                        train_transforms_def=ComposeTransformDefinition(
                            ComposeTransformHyperParameterSet([
                                ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                                RandomHorizontalFlipTransformDefinition(),
                                RandomCropTransformDefinition(
                                    RandomCropTransformHyperParameterSet(size=224, padding=16)),
                                ToTensorTransformDefinition(),
                                RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                                NormalizeTransformDefinition(
                                    NormalizeHyperParameterSet(mean=[0.5118, 0.4911, 0.4646],
                                                               std=[0.3352, 0.3299, 0.3384]))
                            ])),
                        val_transforms_def=None,
                        test_transforms_def=ComposeTransformDefinition(
                            ComposeTransformHyperParameterSet([
                                ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                                ToTensorTransformDefinition(),
                                RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                                NormalizeTransformDefinition(
                                    NormalizeHyperParameterSet(mean=[0.5118, 0.4911, 0.4646],
                                                               std=[0.3352, 0.3299, 0.3384]))
                            ]))))
            }
        return data_definitions[self.arguments.data_type]

    def _get_model_definition(self, num_classes):
        optimizer_definition = self._get_optimizer_definition()
        scheduler_definition = MultiStepLRSchedulerDefinition(
            MultiStepLRSchedulerHyperParameterSet(
                max_epochs=self.arguments.max_epochs,
                milestone_ratios=self.arguments.lr_milestone_ratio,
                gamma=self.arguments.lr_milestone_gamma
            )
        )
        loss_calc_params = self._get_loss_calc_params(num_classes)

        if self.arguments.backbone_model == ModelType.EfficientNetB3:
            backbone_size = 1536
        elif self.arguments.backbone_model == ModelType.DenseNet:
            backbone_size = 1024
        else:
            backbone_size = 512

        model_definitions = \
            {
                ModelType.ResNet18: self._update_iid_loss_calc_params(
                    self._update_gradient_multiplier(
                        self._get_backbone_def(
                            ModelType.ResNet18,
                            self.arguments.pretraining_type,
                            num_classes,
                            optimizer_definition,
                            scheduler_definition,
                            loss_calc_params))),
                ModelType.ResNet34: self._update_iid_loss_calc_params(
                    self._update_gradient_multiplier(
                        self._get_backbone_def(
                            ModelType.ResNet34,
                            self.arguments.pretraining_type,
                            num_classes,
                            optimizer_definition,
                            scheduler_definition,
                            loss_calc_params))),
                ModelType.ResNet50: self._update_iid_loss_calc_params(
                    self._update_gradient_multiplier(
                        self._get_backbone_def(
                            ModelType.ResNet50,
                            self.arguments.pretraining_type,
                            num_classes,
                            optimizer_definition,
                            scheduler_definition,
                            loss_calc_params))),
                ModelType.VGG11: self._update_iid_loss_calc_params(
                    self._update_gradient_multiplier(
                        self._get_backbone_def(
                            ModelType.VGG11,
                            PretrainingType.NONE,
                            num_classes,
                            optimizer_definition,
                            scheduler_definition,
                            loss_calc_params))),
                ModelType.EfficientNetB3: self._update_iid_loss_calc_params(
                    self._update_gradient_multiplier(
                        self._get_backbone_def(
                            ModelType.EfficientNetB3,
                            PretrainingType.NONE,
                            num_classes,
                            optimizer_definition,
                            scheduler_definition,
                            loss_calc_params))),
                ModelType.WideResNet: self._update_iid_loss_calc_params(
                    self._update_gradient_multiplier(
                        self._get_backbone_def(
                            ModelType.WideResNet,
                            PretrainingType.NONE,
                            num_classes,
                            optimizer_definition,
                            scheduler_definition,
                            loss_calc_params))),
                ModelType.DenseNet: self._update_iid_loss_calc_params(
                    self._update_gradient_multiplier(
                        self._get_backbone_def(
                            ModelType.DenseNet,
                            self.arguments.pretraining_type,
                            num_classes,
                            optimizer_definition,
                            scheduler_definition,
                            loss_calc_params))),
                ModelType.FeatureRefiner: FeatureRefinerDefinition(
                    FeatureRefinerHyperParameterSet(
                        iid_net_def=self._update_iid_loss_calc_params(
                            self._get_backbone_def(
                                self.arguments.backbone_model,
                                self.arguments.pretraining_type,
                                num_classes,
                                optimizer_definition,
                                scheduler_definition,
                                loss_calc_params)),
                        fr_net_def=FRNetDefinition(
                            FRNetHyperParameterSet(
                                backbone_size=backbone_size,
                                feature_size=self.arguments.feature_size,
                                output_size=num_classes,
                                num_inner_layers=self.arguments.num_inner_layers,
                                head_bias=self.arguments.head_bias,
                                disable_norm1=self.arguments.disable_norm1,
                                disable_norm2=self.arguments.disable_norm2,
                                disable_layer1=self.arguments.disable_layer1,
                                disable_layer2=self.arguments.disable_layer2,
                                optimizer_definition=None,
                                scheduler_definition=None
                            )),
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    )),
                ModelType.SimpleFeatureRefiner: FeatureRefinerDefinition(
                    FeatureRefinerHyperParameterSet(
                        iid_net_def=self._update_iid_loss_calc_params(
                            self._get_backbone_def(
                                self.arguments.backbone_model,
                                self.arguments.pretraining_type,
                                num_classes,
                                optimizer_definition,
                                scheduler_definition,
                                loss_calc_params)),
                        fr_net_def=SimpleFRNetDefinition(
                            SimpleFRNetHyperParameterSet(
                                backbone_size=backbone_size,
                                feature_size=self.arguments.feature_size,
                                output_size=num_classes,
                                num_inner_layers=self.arguments.num_inner_layers,
                                head_bias=self.arguments.head_bias,
                                optimizer_definition=None,
                                scheduler_definition=None
                            )),
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    )),
                ModelType.DMLNet: DMLNetDefinition(
                    DMLNetHyperParameterSet(
                        main_net_def=self._update_iid_loss_calc_params(
                            self._update_gradient_multiplier(
                                self._get_backbone_def(
                                    self.arguments.backbone_model,
                                    self.arguments.pretraining_type,
                                    num_classes,
                                    optimizer_definition,
                                    scheduler_definition,
                                    loss_calc_params))),
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    )),
                ModelType.KDCLNet: KDCLNetDefinition(
                    KDCLNetHyperParameterSet(
                        main_net_def=self._update_iid_loss_calc_params(
                            self._update_gradient_multiplier(
                                self._get_backbone_def(
                                    self.arguments.backbone_model,
                                    self.arguments.pretraining_type,
                                    num_classes,
                                    optimizer_definition,
                                    scheduler_definition,
                                    loss_calc_params))),
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    )),
                ModelType.KDNet: KDNetDefinition(
                    KDNetHyperParameterSet(
                        main_net_def=self._update_iid_loss_calc_params(
                            self._update_gradient_multiplier(
                                self._get_backbone_def(
                                    self.arguments.backbone_model,
                                    self.arguments.pretraining_type,
                                    num_classes,
                                    optimizer_definition,
                                    scheduler_definition,
                                    loss_calc_params))),
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    )),
                ModelType.Dummy: DummyModuleDefinition(
                    DummyModuleHyperParameterSet(
                        output_size=num_classes,
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params))
            }

        if self.arguments.trainer_type == TrainerType.LearningLossTrainer:
            if self.arguments.pretraining_type != PretrainingType.NONE or self.arguments.data_type in (
            DataModuleType.CALTECH101, DataModuleType.CALTECH256):
                feature_sizes = [56, 28, 14, 7]
            else:
                feature_sizes = [32, 16, 8, 4]
            return LearningLossDefinition(
                LearningLossHyperParameterSet(
                    loss_net_def=LossNetDefinition(
                        LossNetHyperParameterSet(
                            feature_sizes=feature_sizes,
                            optimizer_definition=optimizer_definition,
                            scheduler_definition=scheduler_definition,
                        )
                    ),
                    main_model_def=model_definitions[self.arguments.model]
                )
            )
        else:
            return model_definitions[self.arguments.model]

    def _get_optimizer_definition(self):
        optimizer_definitions = {
            OptimizerType.SGD: SGDOptimizerDefinition(
                SGDOptimizerHyperParameterSet(
                    lr=self.arguments.learning_rate,
                    momentum=self.arguments.momentum,
                    weight_decay=self.arguments.weight_decay)),
            OptimizerType.Adam: AdamOptimizerDefinition(
                AdamOptimizerHyperParameterSet(
                    lr=self.arguments.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=self.arguments.weight_decay,
                    amsgrad=False
                )
            ),
            OptimizerType.RAdam: RAdamOptimizerDefinition(
                RAdamOptimizerHyperParameterSet(
                    lr=self.arguments.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=self.arguments.weight_decay
                )
            )
        }
        return optimizer_definitions[self.arguments.optimizer_type]

    def _get_trainer_definition(self):
        if self.arguments.validate_epochs:
            check_val_every_n_epoch = self.arguments.validate_epochs
        else:
            check_val_every_n_epoch = self.arguments.max_epochs + 1

        trainer_definitions = {
            TrainerType.PL_Trainer: PL_TrainerDefinition(
                PL_TrainerHyperParameterSet(
                    runtime_mode=self.arguments.device,
                    max_epochs=self.arguments.max_epochs,
                    fast_dev_run=self.arguments.fast_dev_run,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    weights_summary=None,
                    checkpoint_callback=True,
                    track_grad_norm=2 if self.arguments.track_grad_norm else -1)),
            TrainerType.LearningLossTrainer: LearningLossTrainerDefinition(
                LearningLossTrainerHyperParameterSet(
                    device=self.arguments.device,
                    max_epochs=self.arguments.max_epochs,
                    lossnet_sheduling=120,
                    lossnet_weight=1.0,
                    lossnet_margin=1.0
                ),
            ),
            TrainerType.FixMatchTrainer: FixMatchTrainerDefinition(
                FixMatchTrainerHyperParameterSet(
                    device=self.arguments.device,
                    max_epochs=self.arguments.max_epochs,
                )
            )
        }

        return trainer_definitions[self.arguments.trainer_type]

    def _get_loss_calc_params(self, num_classes=None):
        all_loss_params = \
            {
                "cross_entropy":
                    LossEvaluatorHyperParameterSet(
                        layers_needed=None,
                        loss_definition=CrossEntropyDefinition()),
                "mpn_cross_entropy":
                    LossEvaluatorHyperParameterSet(
                        layers_needed=None,
                        loss_definition=CrossEntropyDefinition()),
                "iid_cross_entropy":
                    LossEvaluatorHyperParameterSet(
                        layers_needed={"input": "model.iid_net.model.classifier"},
                        loss_definition=CrossEntropyDefinition()),
                "center_loss":
                    LossEvaluatorHyperParameterSet(
                        layers_needed={"x": "flatten4"},
                        loss_definition=CenterLossDefinition(
                            CenterLossHyperParameterSet(
                                num_classes=num_classes,
                                feat_dim=512
                            )
                        )),
                "batch_center_loss":
                    LossEvaluatorHyperParameterSet(
                        layers_needed={"x": "flatten4"},
                        loss_definition=BatchCenterLossDefinition(
                            BatchCenterLossHyperParameterSet(
                                distance_metric=self._get_distance_metric(self.arguments.batch_center_loss_metric)
                            )
                        )),
                "mpn_center_loss":
                    LossEvaluatorHyperParameterSet(
                        layers_needed={"x": "model.mpn_net.model.mpn_layers.layer_0"},
                        loss_definition=CenterLossDefinition(
                            CenterLossHyperParameterSet(
                                num_classes=num_classes,
                                feat_dim=self.arguments.feature_size
                            )
                        )),
                "mpn_batch_center_loss":
                    LossEvaluatorHyperParameterSet(
                        layers_needed={"x": "model.mpn_net.model.mpn_layers.layer_0"},
                        loss_definition=BatchCenterLossDefinition(
                            BatchCenterLossHyperParameterSet(
                                distance_metric=self._get_distance_metric(self.arguments.batch_center_loss_metric)
                            )
                        )),
                "iid_center_loss":
                    LossEvaluatorHyperParameterSet(
                        layers_needed={"x": "model.iid_net.model.backbone.flatten"},
                        loss_definition=CenterLossDefinition(
                            CenterLossHyperParameterSet(
                                num_classes=num_classes,
                                feat_dim=512
                            )
                        )),
                "dml_loss":
                    LossEvaluatorHyperParameterSet(
                        layers_needed=None,
                        loss_definition=DMLLossDefinition()),
                "kdcl_loss":
                    LossEvaluatorHyperParameterSet(
                        layers_needed=None,
                        loss_definition=KDCLLossDefinition()),
                "kd_loss":
                    LossEvaluatorHyperParameterSet(
                        layers_needed=None,
                        loss_definition=KDLossDefinition()),
            }

        if self.arguments.loss_definition is None:
            if self.arguments.model in (ModelType.GeneralNet, ModelType.FeatureRefiner):
                loss_definition = ["mpn_cross_entropy", "iid_cross_entropy"]
            elif self.arguments.model in (ModelType.DMLNet,):
                loss_definition = ["dml_loss"]
            elif self.arguments.model in (ModelType.KDCLNet,):
                loss_definition = ["kdcl_loss"]
            elif self.arguments.model in (ModelType.KDNet,):
                loss_definition = ["kd_loss"]
            else:
                loss_definition = ["cross_entropy"]
        else:
            loss_definition = self.arguments.loss_definition

        loss_calc_params = {}
        if self.arguments.loss_weights is not None:
            assert len(self.arguments.loss_weights) == len(loss_definition), \
                "The size of the loss weights does not match the size of the loss definitions!"
            loss_weights = self.arguments.loss_weights
        else:
            loss_weights = [1] * len(loss_definition)

        for loss_weight, loss_param_id in zip(loss_weights, loss_definition):
            loss_calc_params[loss_param_id] = all_loss_params[loss_param_id]
            loss_calc_params[loss_param_id].weight = loss_weight

        return loss_calc_params

    def _update_loss_feature_size(self, optimization_definition_set):
        if optimization_definition_set.model_definition.type == ModelType.GeneralNet:
            for loss_param in optimization_definition_set.model_definition.hyperparams.loss_calc_params.values():
                loss_hyperparams = loss_param.loss_definition.hyperparams
                if hasattr(loss_hyperparams, "feat_dim"):
                    loss_hyperparams.feat_dim = \
                        optimization_definition_set.model_definition.hyperparams.mpn_net_def.hyperparams.feature_size
        if optimization_definition_set.model_definition.type == ModelType.FeatureRefiner:
            for loss_param in optimization_definition_set.model_definition.hyperparams.loss_calc_params.values():
                loss_hyperparams = loss_param.loss_definition.hyperparams
                if hasattr(loss_hyperparams, "feat_dim"):
                    loss_hyperparams.feat_dim = \
                        optimization_definition_set.model_definition.hyperparams.fr_net_def.hyperparams.feature_size

    @staticmethod
    def _update_iid_loss_calc_params(iid_model_def):
        updated_iid_model_def = copy.deepcopy(iid_model_def)
        updated_iid_model_def.hyperparams.loss_calc_params = \
            {key: params for key, params in updated_iid_model_def.hyperparams.loss_calc_params.items() if "iid" in key}
        for param in updated_iid_model_def.hyperparams.loss_calc_params.values():
            if param.layers_needed is not None:
                param.layers_needed = {key: remove_prefix(layer_name, "model.iid_net.")
                                       for key, layer_name in param.layers_needed.items()}
        updated_iid_model_def.hyperparams.loss_calc_params.update(
            {key: params for key, params in iid_model_def.hyperparams.loss_calc_params.items() if
             "iid" not in key and "mpn" not in key})
        return updated_iid_model_def

    def _update_gradient_multiplier(self, resnet_def):
        resnet_def = copy.deepcopy(resnet_def)
        resnet_def.hyperparams.gradient_multiplier = 1.0
        return resnet_def

    def _update_default_params(self):
        if self.arguments.batch_size is None:
            self.arguments.batch_size = 100

        if self.arguments.feature_size is None:
            self.arguments.feature_size = 50

        if self.arguments.num_message_passings is None:
            self.arguments.num_message_passings = 1

        if self.arguments.num_heads is None:
            self.arguments.num_heads = 4

        if self.arguments.learning_rate is None:
            self.arguments.learning_rate = 0.1

        if self.arguments.lr_milestone_ratio is None:
            self.arguments.lr_milestone_ratio = [0.8]

        if self.arguments.lr_milestone_gamma is None:
            self.arguments.lr_milestone_gamma = 0.1

        if self.arguments.momentum is None:
            self.arguments.momentum = 0.9

        if self.arguments.weight_decay is None:
            self.arguments.weight_decay = 5e-4

        if self.arguments.max_epochs is None:
            self.arguments.max_epochs = 200

    def _get_fixed_params(self):
        params = {
            "batch_size": self.arguments.batch_size,
            "feature_size": self.arguments.feature_size,
            "num_message_passings": self.arguments.num_message_passings,
            "num_heads": self.arguments.num_heads,
            "attention_scaling_threshold": self.arguments.attention_scaling_threshold,
            "learning_rate": self.arguments.learning_rate,
            "lr_milestone_ratio": self.arguments.lr_milestone_ratio,
            "momentum": self.arguments.momentum,
            "weight_decay": self.arguments.weight_decay,
            "max_epochs": self.arguments.max_epochs
        }

        return {key: value for key, value in params.items() if value is not None}

    def _get_description(self):
        fixed_params = self._get_fixed_params()
        if len(fixed_params) == 0:
            return None

        return "|".join(f"{key}-{value}" for key, value in fixed_params.items())

    def _update_intermediate_layers_def(self, iid_def):
        if iid_def.type in (ModelType.ResNet18, ModelType.ResNet34, ModelType.ResNet50):
            iid_def = copy.deepcopy(iid_def)
            iid_def.hyperparams.backbone_definition.hyperparams.intermediate_layers_to_return = self.arguments.intermediate_layers
        return iid_def

    def _get_intermediate_sizes(self):
        resnet_intermediate_sizes = {
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512
        }
        return {layer_name: resnet_intermediate_sizes[layer_name] for layer_name in self.arguments.intermediate_layers}

    def _get_distance_metric(self, metric_type):
        metric_defs = {
            DistanceType.EUCLIDEAN_DISTANCE: EuclideanDistanceDefinitionSet(),
            DistanceType.COSINE_DISTANCE: CosineDistanceDefinitionSet(),
            DistanceType.CORRELATION_DISTANCE: CorrelationDistanceDefinitionSet()
        }
        return metric_defs[metric_type]

    def _get_backbone_def(self, model, pretraining_type, num_classes, optimizer_definition, scheduler_definition,
                          loss_calc_params):
        if pretraining_type != PretrainingType.NONE:
            assert model not in (ModelType.VGG11, ModelType.EfficientNetB3), "VGG pretrained is not supported yet!"
            if model in (ModelType.ResNet50, ModelType.ResNet34, ModelType.ResNet18):
                resnet_types = {
                    ModelType.ResNet18: ResNetType.ResNet18,
                    ModelType.ResNet34: ResNetType.ResNet34,
                    ModelType.ResNet50: ResNetType.ResNet50
                }
                resnet_type = resnet_types[model]
                iid_def = PretrainedResNetDefinition(
                    PretrainedResNetHyperParameterSet(
                        type=resnet_type,
                        pretraining_type=pretraining_type,
                        output_size=num_classes,
                        gradient_multiplier=self.arguments.gradient_multiplier,
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    ))
            elif model == ModelType.DenseNet:
                iid_def = PretrainedDenseNetDefinition(
                    PretrainedDenseNetHyperParameterSet(
                        pretraining_type=pretraining_type,
                        output_size=num_classes,
                        gradient_multiplier=self.arguments.gradient_multiplier,
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    ))
        else:
            if model == ModelType.ResNet34:
                iid_def = ResNet34Definition(
                    ResNet34HyperParameterSet(
                        output_size=num_classes,
                        head_bias=self.arguments.head_bias,
                        gradient_multiplier=self.arguments.gradient_multiplier,
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    ))
            elif model == ModelType.ResNet50:
                iid_def = ResNet50Definition(
                    ResNet50HyperParameterSet(
                        output_size=num_classes,
                        head_bias=self.arguments.head_bias,
                        gradient_multiplier=self.arguments.gradient_multiplier,
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    ))
            elif model == ModelType.VGG11:
                iid_def = VGG11Definition(
                    VGG11HyperParameterSet(
                        feature_size=4096,
                        gradient_multiplier=self.arguments.gradient_multiplier,
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    )
                )
            elif model == ModelType.EfficientNetB3:
                iid_def = EfficientNetB3Definition(
                    EfficientNetB3HyperParameterSet(
                        output_size=num_classes,
                        gradient_multiplier=self.arguments.gradient_multiplier,
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    )
                )
            elif model == ModelType.WideResNet:
                iid_def = WideResNetDefinition(
                    WideResNetHyperParameterSet(
                        output_size=num_classes,
                        gradient_multiplier=self.arguments.gradient_multiplier,
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    )
                )
            elif model == ModelType.DenseNet:
                iid_def = DenseNetDefinition(
                    DenseNetHyperParameterSet(
                        output_size=num_classes,
                        gradient_multiplier=self.arguments.gradient_multiplier,
                        optimizer_definition=optimizer_definition,
                        scheduler_definition=scheduler_definition,
                        loss_calc_params=loss_calc_params
                    )
                )
            else:
                if self.arguments.data_type in (DataModuleType.CALTECH101, DataModuleType.CALTECH256):
                    initial_kernel_reduced = False
                else:
                    initial_kernel_reduced = True

                if model == ModelType.ResNet18:
                    iid_def = ResNet18Definition(
                        ResNet18HyperParameterSet(
                            output_size=num_classes,
                            head_bias=self.arguments.head_bias,
                            gradient_multiplier=self.arguments.gradient_multiplier,
                            initial_kernel_reduced=initial_kernel_reduced,
                            optimizer_definition=optimizer_definition,
                            scheduler_definition=scheduler_definition,
                            loss_calc_params=loss_calc_params
                        ))

        iid_def = self._update_intermediate_layers_def(iid_def)

        if self.arguments.trainer_type == TrainerType.LearningLossTrainer and self.arguments.pretraining_type == PretrainingType.NONE:
            if hasattr(iid_def.hyperparams, "backbone_definition"):
                iid_def.hyperparams.backbone_definition.hyperparams.intermediate_layers_to_return = ["layer1", "layer2",
                                                                                                     "layer3", "layer4"]
        return iid_def


if __name__ == "__main__":
    arguments = ActiveLearningExperiment.argparser().parse_args()
    experiment = ActiveLearningExperiment.from_argparse_args(arguments)
    experiment.run()
