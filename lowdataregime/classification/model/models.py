from abc import ABC

from lowdataregime.classification.model.classification_module import ClassificationModuleHyperParameterSet
from lowdataregime.parameters.params import DefinitionSet
from lowdataregime.utils.utils import SerializableEnum


class PretrainingType(SerializableEnum):
    NONE = "none"
    ImageNet = "imagenet"


class ModelType(SerializableEnum):
    """Definition of the available models"""
    ResNet18Backbone = "ResNet18Backbone"
    ResNet34Backbone = "ResNet34Backbone"
    ResNet50Backbone = "ResNet50Backbone"
    ResNet18 = "ResNet18"
    ResNet34 = "ResNet34"
    ResNet50 = "ResNet50"
    ResNet101 = "ResNet101"
    ResNet152 = "ResNet152"
    PretrainedResNet18 = "PretrainedResNet18"
    Dummy = "dummy"
    MessagePassingNet = "MPN"
    FRNet = "FRNet"
    SimpleFRNet = "SimpleFRNet"

    FeatureRefiner = "FeatureRefiner"
    SimpleFeatureRefiner = "SimpleFeatureRefiner"
    GeneralNet = "GeneralNet"
    JennyNet = "JennyNet"

    LearningLoss = "LearningLoss"

    ResNet18red = "ResNet18red"

    VGG11Backbone = "VGG11Backbone"
    VGG11 = "VGG11"

    EfficientNetB3Backbone = "EfficientNetB3Backbone"
    EfficientNetB3 = "EfficientNetB3"

    ResNet18BlockRed = "ResNet18BlockRed"

    ResNet18Small = "ResNet18Small"

    WideResNetBackbone = "WideResNetBackbone"
    WideResNet = "WideResNet"

    DenseNetBackbone = "DenseNetBackbone"
    DenseNet = "DenseNet"
    PretrainedDenseNet = "PretrainedDenseNet"

    DMLNet = "DMLNet"
    KDCLNet = "KDCLNet"
    KDNet = "KDNet"


class ClassificationModuleDefinition(DefinitionSet, ABC):
    """Abstract definition of a ClassificationModule"""

    def __init__(self, type: ModelType = None, hyperparams: ClassificationModuleHyperParameterSet = None):
        super().__init__(type, hyperparams)
