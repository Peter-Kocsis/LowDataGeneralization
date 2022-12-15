import os
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from lowdataregime.active_learning.query.uncertainty_query import UncertaintyQueryDefinition, UncertaintyQueryHyperParameterSet
from lowdataregime.classification.data.cifar import CIFARDataModule, CIFARHyperParameterSet
from lowdataregime.classification.data.datas import DataModuleType
from lowdataregime.classification.sampling.sampler import SubsetSequentialSampler
from lowdataregime.classification.data.transform import ToTensorTransformDefinition, ComposeTransformDefinition, \
    ComposeTransformHyperParameterSet, NormalizeTransformDefinition, NormalizeHyperParameterSet
from lowdataregime.classification.model.models import ModelType
from lowdataregime.classification.trainer.trainers import PL_TrainerDefinition
from lowdataregime.parameters.params import HyperParameterSet, DefinitionSet
from lowdataregime.utils.utils import Serializable
from lowdataregime.parameters.active_loader import ActiveLoaderDefinition


class ActiveVisualizerHyperParameterSet(HyperParameterSet):

    def __init__(self,
                 experiment_path: str = None,
                 active_loader_def: ActiveLoaderDefinition = ActiveLoaderDefinition(),
                 trainer_definition: PL_TrainerDefinition = PL_TrainerDefinition(),
                 **kwargs):
        super().__init__(**kwargs)
        self.experiment_path = experiment_path
        self.active_loader_def = active_loader_def
        self.trainer_definition = trainer_definition


class ActiveVisualizerDefinition(DefinitionSet):

    def __init__(self,
                 hyperparams: ActiveVisualizerHyperParameterSet = ActiveVisualizerHyperParameterSet()):
        super().__init__(hyperparams=hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        # TODO: Make it cleaner -> maybe remove
        return None

    def instantiate(self, experiment, *args, **kwargs):
        """Instantiates the module"""
        return ActiveVisualizer(params=self.hyperparams)


class ActiveVisualizer:

    def __init__(self, params: ActiveVisualizerHyperParameterSet = ActiveVisualizerHyperParameterSet()):
        self.params = params
        self.experiment_path = params.experiment_path
        self.active_loader = params.active_loader_def.instantiate()

        self.labeled_predictions = None
        self.labeled_labels = None
        self.unlabeled_labels = None
        self.labeled_batches = None
        self.unlabeled_predictions = None
        self.unlabeled_batches = None
        self.labeled_uncertainties = None
        self.unlabeled_uncertainties = None

        self.datamodules = None
        self.models = None
        self.raw_data = None

    @staticmethod
    def _get_stage_id(current_stage: int):
        return f"stage_{current_stage}"

    def _get_logger_folder_path(self, stage):
        return os.path.join(self.experiment_path, self._get_stage_id(stage))

    def _get_model(self, stage: int, memorize: bool = False):
        if self.models is not None:
            if stage in self.models:
                model = self.models[stage]
            else:
                model = self.active_loader.load_model(stage)
        else:
            model = self.active_loader.load_model(stage)
        if memorize:
            if self.models is None:
                self.models = {stage: model}
            else:
                self.models[stage] = model
        return model

    def _get_datamodule(self, stage: int, memorize: bool = False):
        if self.datamodules is not None:
            if stage in self.datamodules:
                datamodule = self.datamodules[stage]
            else:
                datamodule = self.active_loader.load_active_datamodule(stage)
        else:
            datamodule = self.active_loader.load_active_datamodule(stage)
        if memorize:
            if self.datamodules is None:
                self.datamodules = {stage: datamodule}
            else:
                self.datamodules[stage] = datamodule
        return datamodule

    def _get_raw_data(self):
        raw_data = CIFARDataModule(
            CIFARHyperParameterSet(
                dataset_to_use=DataModuleType.CIFAR10,
                num_workers=0,
                batch_size=100,
                val_ratio=0.0,
                duplication_factor=1,
                train_transforms_def=ComposeTransformDefinition(
                    ComposeTransformHyperParameterSet([
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
                            ]))))
        return raw_data

    def _get_training_params(self):
        # return ActiveLearningTrainingLog(self.experiment_path).get_params()
        return Serializable.loads_from_file(os.path.join(self.experiment_path, "active_learning_params.json"))

    def _create_logger(self, stage):  # TODO
        pass
        '''
        ClassificationLogger(save_dir=self._get_logger_folder_path(),
                             name=os.path.join(self.training_scope, self.training_name),
                             version=self.status.get_stage_id(stage))
        '''

    def print_easter_egg(self):
        model = self._get_model(stage=0)
        print(model.return_easter_egg())

    def evaluate_test_accuracy(self, stage: int):
        model = self._get_model(stage=stage)
        model.eval()
        datamodule = self._get_datamodule(stage=stage)
        model.init_metrics()
        trainer = self.params.trainer_definition.instantiate()
        trainer.test(model=model, datamodule=datamodule.dataset)
        acc = model.test_acc.compute()
        print(acc.cpu().numpy().tolist())

    def obtain_predictions_batches_and_labels_on_labeled_pool(self, stage: int, memorize: bool = False):
        model = self._get_model(stage=stage)
        model.eval()
        datamodule = self._get_datamodule(stage=stage)
        training_params = self._get_training_params()
        temperature = training_params.query_definition.hyperparams.temperature
        inference_labeled_size = datamodule.dataset.batch_size
        labeled_loader = DataLoader(
            datamodule.unaugmented_dataset_train, batch_size=inference_labeled_size,
            sampler=SubsetSequentialSampler(datamodule.labeled_pool_indices),
            num_workers=datamodule.num_workers,
            drop_last=True,
            # more convenient if we maintain the order of subset
            pin_memory=True)
        model.eval()
        predictions = torch.tensor([], device=model.device)
        batches = torch.tensor([], device=model.device)
        labels = torch.tensor([], device=model.device)
        iteration = 0
        with torch.no_grad():
            for (labeled_inputs, batch_labels) in tqdm(labeled_loader, leave=False,
                                                       total=len(labeled_loader)):
                labeled_inputs = labeled_inputs.to(model.device)
                batch_labels = batch_labels.to(model.device)

                scores = model(labeled_inputs)
                cool_scores = scores / temperature
                probs = F.softmax(cool_scores, dim=1)
                predictions = torch.cat((predictions, probs), 0)

                batch = torch.tensor(datamodule.labeled_pool_indices[inference_labeled_size * iteration:
                                                                     inference_labeled_size * (iteration + 1)]).to(
                    model.device)
                labels = torch.cat((labels, batch_labels), 0)

                if iteration == 0:
                    batches = batch
                elif iteration == 1:
                    batches = torch.stack((batches, batch), 0)
                else:
                    batches = torch.cat((batches, batch.unsqueeze(1).T), 0)

                iteration += 1

        if memorize:
            if self.labeled_batches is None:
                self.labeled_batches = {stage: batches}
                self.labeled_predictions = {stage: predictions}
                self.labeled_labels = {stage: labels}
            else:
                self.labeled_batches[stage] = batches
                self.labeled_predictions[stage] = predictions
                self.labeled_labels[stage] = labels

        return predictions.cpu().numpy(), batches.cpu().numpy()

    def obtain_uncertainties_on_labeled_pool(self, stage, memorize: bool = False, plot_hist: bool = False):
        assert self.labeled_predictions[stage] is not None, \
            "Cannot obtain uncertainties unless predictions are obtained"

        probs = self.labeled_predictions[stage]
        log_probs = torch.log(probs)
        to_be_summed = probs * log_probs
        uncertainties = torch.t(-torch.sum(to_be_summed, dim=1, keepdim=True))
        if memorize:
            if self.labeled_uncertainties is None:
                self.labeled_uncertainties = {stage: uncertainties}
            else:
                self.labeled_uncertainties[stage] = uncertainties
        if plot_hist:
            plt.hist(uncertainties)
            plt.show()

        return uncertainties

    def obtain_predictions_batches_and_labels_on_unlabeled_pool(self, stage: int, memorize: bool = False):
        model = self._get_model(stage=stage)
        model.eval()
        datamodule = self._get_datamodule(stage=stage)
        training_params = self._get_training_params()
        temperature = training_params.query_definition.hyperparams.temperature
        inference_unlabeled_size = datamodule.dataset.batch_size
        unlabeled_loader = DataLoader(
            datamodule.unaugmented_dataset_train, batch_size=inference_unlabeled_size,
            sampler=SubsetSequentialSampler(datamodule.unlabeled_pool_indices),
            num_workers=datamodule.num_workers,
            drop_last=True,
            # more convenient if we maintain the order of subset
            pin_memory=True)
        model.eval()
        predictions = torch.tensor([], device=model.device)
        batches = torch.tensor([], device=model.device)
        labels = torch.tensor([], device=model.device)
        iteration = 0
        with torch.no_grad():
            for (unlabeled_inputs, batch_labels) in tqdm(unlabeled_loader, leave=False,
                                                         total=len(unlabeled_loader)):
                unlabeled_inputs = unlabeled_inputs.to(model.device)
                batch_labels = batch_labels.to(model.device)

                scores = model(unlabeled_inputs)
                cool_scores = scores / temperature
                probs = F.softmax(cool_scores, dim=1)
                predictions = torch.cat((predictions, probs), 0)

                batch = torch.tensor(datamodule.unlabeled_pool_indices[inference_unlabeled_size * iteration:
                                                                       inference_unlabeled_size * (iteration + 1)]).to(
                    model.device)
                labels = torch.cat((labels, batch_labels), 0)

                if iteration == 0:
                    batches = batch
                elif iteration == 1:
                    batches = torch.stack((batches, batch), 0)
                else:
                    batches = torch.cat((batches, batch.unsqueeze(1).T), 0)

                iteration += 1

        if memorize:
            if self.unlabeled_batches is None:
                self.unlabeled_batches = {stage: batches}
                self.unlabeled_predictions = {stage: predictions}
                self.unlabeled_labels = {stage: labels}
            else:
                self.unlabeled_batches[stage] = batches
                self.unlabeled_predictions[stage] = predictions
                self.unlabeled_labels[stage] = labels

        return predictions.cpu().numpy(), batches.cpu().numpy()

    def obtain_uncertainties_on_unlabeled_pool(self, stage, memorize: bool = False, plot_hist: bool = False):
        assert self.unlabeled_predictions[stage] is not None, \
            "Cannot obtain uncertainties unless predictions are obtained"

        probs = self.unlabeled_predictions[stage]
        log_probs = torch.log(probs)
        to_be_summed = probs * log_probs
        uncertainties = torch.t(-torch.sum(to_be_summed, dim=1, keepdim=True))
        if memorize:
            if self.unlabeled_uncertainties is None:
                self.unlabeled_uncertainties = {stage: uncertainties}
            else:
                self.unlabeled_uncertainties[stage] = uncertainties
        if plot_hist:
            plt.hist(uncertainties)
            plt.show()

        return uncertainties

    def obtain_labeled_pool_class_distribution(self, stage, by_batch: bool = False):
        assert self.labeled_labels[stage] is not None, \
            "Cannot obtain class distribution unless labels are obtained"

        if not by_batch:
            unique, counts = np.unique(self.labeled_labels[stage], return_counts=True)
            return dict(zip(unique, counts))

        if by_batch:
            num_classes = self._get_training_params().optimization_definition_set.data_definition.hyperparams.num_classes
            batch_size = self._get_training_params().optimization_definition_set.data_definition.hyperparams.batch_size
            labels = self.labeled_labels[stage].reshape((-1, batch_size))
            class_counts = np.zeros((labels.shape[0], num_classes))
            for i in range(num_classes):
                class_counts[:, i] = (labels == i).sum(axis=1)
            return class_counts

    def obtain_unlabeled_pool_class_distribution(self, stage, by_batch: bool = False):
        assert self.unlabeled_labels[stage] is not None, \
            "Cannot obtain class distribution unless labels are obtained"

        if not by_batch:
            unique, counts = np.unique(self.unlabeled_labels[stage], return_counts=True)
            return dict(zip(unique, counts))

        if by_batch:
            num_classes = self._get_training_params().optimization_definition_set.data_definition.hyperparams.num_classes
            batch_size = self._get_training_params().optimization_definition_set.data_definition.hyperparams.batch_size
            labels = self.unlabeled_labels[stage].reshape((-1, batch_size))
            class_counts = np.zeros((labels.shape[0], num_classes))
            for i in range(num_classes):
                class_counts[:, i] = (labels == i).sum(axis=1)
            return class_counts

    def obtain_labeling_class_distribution(self, stage):
        assert self.labeled_labels[stage + 1] is not None, \
            "Cannot obtain labeling set unless labels are obtained"

        labeling_size = self._get_training_params().num_new_labeled_samples_per_stage
        labels = self.labeled_labels[stage + 1][labeling_size * (stage + 1):]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    def obtain_average_attention(self, stage, num_samples_to_evaluate):
        attention_query_definition = AttentionQueryDefinition()
        attention_query = attention_query_definition.instantiate(".")

        model = self._get_model(stage=stage, memorize=False)
        model.eval()
        datamodule = self._get_datamodule(stage=stage)

        average_attentions = attention_query.evaluate_metric(model, datamodule, num_samples_to_evaluate)
        return average_attentions

    def obtain_uncertaintes(self, stage, num_samples_to_evaluate):
        model = self._get_model(stage=stage, memorize=False)
        model.eval()
        datamodule = self._get_datamodule(stage=stage)
        training_params = self._get_training_params()

        uncertainty_query_definition = UncertaintyQueryDefinition(
            UncertaintyQueryHyperParameterSet(temperature=training_params.query_definition.hyperparams.temperature)
        )
        uncertainty_query = uncertainty_query_definition.instantiate(".")

        uncertainties = uncertainty_query.evaluate_metric(model, datamodule, num_samples_to_evaluate)
        return uncertainties


    def visualize_layers_tsne(self, stage, indices, layers_to_visualize: List[str]):
        inspected_variables, targets = self.inspect_model_on_custom_batch(stage, indices, layers_to_visualize)
        targets = targets.cpu().numpy().astype(int)

        colors = np.array(px.colors.qualitative.Plotly)

        for layer_name, inspected_batch_variable in inspected_variables.items():
            for batch_idx, inspected_variable in enumerate(inspected_batch_variable):
                inspected_variable = inspected_variable.cpu().detach().numpy()

                tsne = TSNE(n_components=2)
                pca_iid = tsne.fit_transform(inspected_variable)
                print(f"Layer {layer_name}/{batch_idx} KL divergence: {tsne.kl_divergence_}")

                fig = go.Figure(layout={"title": f"{layer_name}/{batch_idx}"})
                fig.add_trace(go.Scatter(x=pca_iid[:, 0],
                                         y=pca_iid[:, 1],
                                         mode='markers',
                                         marker_color=colors[targets]))
                fig.show()

    def visualize_features_tsne(self, stage, indices):
        _, _, _, _, _, mpn_features, _ = self.evaluate_model_on_custom_batch(
            stage=stage, indices=indices, distinct_graph=False)
        print("Evaluated with MPN")
        _, _, _, _, _, iid_features, targets = self.evaluate_model_on_custom_batch(
            stage=stage, indices=indices, distinct_graph=True)
        print("Evaluated without MPN")

        difference = mpn_features.numpy() - iid_features.numpy()
        print(f"Maximum difference {np.max(difference)}")

        targets = targets.cpu().numpy().astype(int)

        tsne = TSNE(n_components=2)
        pca_iid = tsne.fit_transform(iid_features)
        print(tsne.kl_divergence_)

        tsne = TSNE(n_components=2)
        pca_mpn = tsne.fit_transform(mpn_features)
        print(tsne.kl_divergence_)

        colors = np.array(px.colors.qualitative.Plotly)

        mpn_refinement_vector = pca_mpn - pca_iid
        fig = make_subplots(rows=1, cols=2, subplot_titles=("IID_features", "MPN_features"))

        # fig = go.Figure()
        fig.add_trace(go.Scatter(x=pca_iid[:, 0],
                                 y=pca_iid[:, 1],
                                 mode='markers',
                                 marker_symbol='x',
                                 marker_color=colors[targets]),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=pca_mpn[:, 0],
                                 y=pca_mpn[:, 1],
                                 mode='markers',
                                 marker_color=colors[targets]),
                      row=1, col=2)
        fig.show()
        print("Features visualized!")

    def visualize_features_pca(self, stage, indices):
        mpn_predictions, mpn_uncertainties, iid_predictions, iid_uncertainties, iid_features, mpn_features, targets = self.evaluate_model_on_custom_batch(
            stage=stage, indices=indices)
        targets = targets.cpu().numpy().astype(int)

        pca = PCA(n_components=10)
        pca.fit(iid_features)

        pca_iid = pca.transform(iid_features)
        pca_mpn = pca.transform(mpn_features)
        print(pca.explained_variance_ratio_)

        # umap_2d = UMAP(n_components=2, init='random', random_state=0)
        # umap_2d.fit(iid_features)
        #
        # pca_iid = umap_2d.transform(iid_features)
        # pca_mpn = umap_2d.transform(mpn_features)

        colors = np.array(px.colors.qualitative.Plotly)

        mpn_refinement_vector = pca_mpn - pca_iid
        fig = ff.create_quiver(pca_iid[:, 0], pca_iid[:, 1], mpn_refinement_vector[:, 0], mpn_refinement_vector[:, 1],
                               scale=1.0,
                               arrow_scale=.1,
                               name='movement',
                               line_width=1)

        # fig = go.Figure()
        fig.add_trace(go.Scatter(x=pca_iid[:, 0],
                                 y=pca_iid[:, 1],
                                 mode='markers',
                                 marker_symbol='x',
                                 marker_color=colors[targets],
                                 name="IID_features"))

        fig.add_trace(go.Scatter(x=pca_mpn[:, 0],
                                 y=pca_mpn[:, 1],
                                 mode='markers',
                                 marker_color=colors[targets],
                                 name="MPN_features"))
        fig.show()
        print("Features visualized!")

    def inspect_model_on_custom_batch(self, stage, indices, layers_to_inspect: List[str], use_test_dataset: bool = False):
        model = self._get_model(stage=stage, memorize=False)
        model.eval()
        datamodule = self._get_datamodule(stage=stage)
        training_params = self._get_training_params()

        batch_size = len(indices)

        if not use_test_dataset:
            unlabeled_loader = DataLoader(
                datamodule.unaugmented_dataset, batch_size=batch_size,
                sampler=SubsetSequentialSampler(indices),
                num_workers=0,
                drop_last=True,
                # more convenient if we maintain the order of subset
                pin_memory=True)
        if use_test_dataset:
            unlabeled_loader = DataLoader(
                datamodule.datamodule.dataset_test, batch_size=batch_size,
                sampler=SubsetSequentialSampler(indices),
                num_workers=0,
                drop_last=True,
                # more convenient if we maintain the order of subset
                pin_memory=True)

        handles = [model.inspect_layer_output(layer_to_inspect)for layer_to_inspect in layers_to_inspect]
        # the_model.model.mpn_net.model.mpn_layers.layer_0.trans.alpha_dropout.register_forward_hook(get_output("alpha"))

        targets = torch.tensor([], device=model.device)
        for (batch, batch_labels) in tqdm(unlabeled_loader, leave=False, total=len(unlabeled_loader)):
            batch = batch.to(model.device)
            batch_output = model(batch)

            targets = torch.cat((targets, batch_labels), 0)

        handles = list(map(lambda handle: handle.remove(), handles))

        return model.inspected_variables, targets

    def evaluate_model_on_custom_batch(self, stage, indices, distinct_graph: bool = False, use_test_set: bool = False,
                                       use_just_skip_layer: bool = False):
        model = self._get_model(stage=stage, memorize=False)
        if distinct_graph:
            model.model.mpn_net.graph_builder = DistinctGraphBuilder(GraphBuilderHyperParameterSet(NoEdgeAttributeDefinitionSet()))
        if use_just_skip_layer:
            model.model.mpn_net.model.mpn_layers.layer_0.trans.__delattr__('lin_beta')
            model.model.mpn_net.model.mpn_layers.layer_0.trans.lin_beta = torch.nn.Linear(3 * model.model.mpn_net.model.mpn_layers.layer_0.trans.out_channels, 1, bias=True)
            model.model.mpn_net.model.mpn_layers.layer_0.trans.lin_beta.weight = torch.nn.parameter.Parameter(torch.zeros((1, 3 * model.model.mpn_net.model.mpn_layers.layer_0.trans.out_channels)))
            model.model.mpn_net.model.mpn_layers.layer_0.trans.lin_beta.bias = torch.nn.parameter.Parameter(torch.Tensor([1]))
        datamodule = self._get_datamodule(stage=stage, memorize=True)
        training_params = self._get_training_params()
        temperature = training_params.query_definition.hyperparams.temperature
        inference_unlabeled_size = datamodule.datamodule.batch_size
        if not use_test_set:
            unlabeled_loader = DataLoader(
                datamodule.unaugmented_dataset, batch_size=len(indices),
                sampler=SubsetSequentialSampler(indices),
                num_workers=0,
                drop_last=True,
                # more convenient if we maintain the order of subset
                pin_memory=True)
        if use_test_set:
            unlabeled_loader = DataLoader(
                datamodule.datamodule.dataset_test, batch_size=len(indices),
                sampler=SubsetSequentialSampler(indices),
                num_workers=0,
                drop_last=True,
                # more convenient if we maintain the order of subset
                pin_memory=True)
        model.eval()
        mpn_predictions = torch.tensor([], device=model.device)
        mpn_uncertainties = torch.tensor([], device=model.device)

        iid_predictions = torch.tensor([], device=model.device)
        iid_uncertainties = torch.tensor([], device=model.device)

        iid_features = torch.tensor([], device=model.device)
        mpn_features = torch.tensor([], device=model.device)

        targets = torch.tensor([], device=model.device)
        with torch.no_grad():
            for (unlabeled_inputs, batch_labels) in tqdm(unlabeled_loader, leave=False,
                                                         total=len(unlabeled_loader)):
                unlabeled_inputs = unlabeled_inputs.to(model.device)
                batch_labels = batch_labels.to(model.device)
                backbone_features, backbone_intermediate = model.model.iid_net.features(unlabeled_inputs)

                iid_feature = model.model.mpn_net.dimension_reduction(backbone_features)
                iid_intermediate = model.model.mpn_net.reduce_intermediate(backbone_intermediate)
                iid_feature = torch.cat((iid_feature, iid_intermediate), dim=1)

                mpn_feature = model.model.mpn_net.feature_refinement(iid_feature)

                mpn_scores = model.model.mpn_net.head(mpn_feature)
                cool_scores = mpn_scores / temperature
                probs = F.softmax(cool_scores, dim=1)
                mpn_predictions = torch.cat((mpn_predictions, probs), 0)
                log_predictions = torch.log(mpn_predictions)
                to_be_summed = probs * log_predictions
                mpn_uncertainty = torch.t(-torch.sum(to_be_summed, dim=1, keepdim=True))
                mpn_uncertainties = torch.cat((mpn_uncertainties, mpn_uncertainty), 0)

                iid_scores = model.model.iid_net.head(backbone_features)
                iid_cool_scores = iid_scores / temperature
                iid_probs = F.softmax(iid_cool_scores, dim=1)
                iid_predictions = torch.cat((iid_predictions, iid_probs), 0)
                iid_log_predictions = torch.log(iid_predictions)
                iid_to_be_summed = iid_probs * iid_log_predictions
                iid_uncertainty = torch.t(-torch.sum(iid_to_be_summed, dim=1, keepdim=True))
                iid_uncertainties = torch.cat((iid_uncertainties, iid_uncertainty), 0)

                iid_features = torch.cat((iid_features, iid_feature), 0)
                mpn_features = torch.cat((mpn_features, mpn_feature), 0)

                targets = torch.cat((targets, batch_labels), 0)

        return backbone_features, iid_scores, iid_predictions, iid_uncertainties, iid_features, mpn_features, mpn_scores, mpn_predictions, mpn_uncertainties, targets

    def show_the_attention_matrices(self, stage, draw_matrices: bool = False):
        the_model = self._get_model(stage=stage)
        the_model.eval()
        training_params = self._get_training_params()
        assert training_params.optimization_definition_set.model_definition.type == ModelType.GeneralNet, \
            "I can work just with MultiClassGeneralNet"
        query_matrix = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_query.weight.cpu().detach().numpy()
        key_matrix = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_key.weight.cpu().detach().numpy()
        value_matrix = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_value.weight.cpu().detach().numpy()
        skip_matrix = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_skip.weight.cpu().detach().numpy()
        query_bias = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_query.bias.cpu().detach().numpy()
        key_bias = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_key.bias.cpu().detach().numpy()
        value_bias = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_value.bias.cpu().detach().numpy()
        skip_bias = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_skip.bias.cpu().detach().numpy()


        '''
        average_gram = np.zeros((50, 50))
        for i in range(4):
            average_gram += query_matrix[i*50:i*50+50, :50] @ key_matrix[i*50:i*50+50, :50].T
        average_gram = average_gram/4
        '''

        if draw_matrices:
            for i in range(
                    training_params.optimization_definition_set.model_definition.hyperparams.mpn_net_def.hyperparams.num_heads):
                fig, ax = plt.subplots()
                ax = ax.matshow(query_matrix[i * 50:i * 50 + 50, :50] @ key_matrix[i * 50:i * 50 + 50, :50].T,
                                cmap=plt.cm.Blues)
                plt.savefig(
                    os.path.join(self.experiment_path, self._get_stage_id(stage), 'gram_matrix_' + str(i) + '.png'))
                plt.title('gram_matrix_' + str(i))
                plt.show()
                plt.close(fig)
                fig, ax = plt.subplots()
                ax = ax.matshow(value_matrix[i * 50:i * 50 + 50, :50], cmap=plt.cm.Blues)
                plt.savefig(
                    os.path.join(self.experiment_path, self._get_stage_id(stage), 'value_matrix_' + str(i) + '.png'))
                plt.title('value_matrix' + str(i))
                plt.show()
                plt.close(fig)
                fig, ax = plt.subplots()
                ax = ax.matshow(query_matrix[i * 50:i * 50 + 50, :50], cmap=plt.cm.Blues)
                plt.savefig(
                    os.path.join(self.experiment_path, self._get_stage_id(stage), 'query_matrix' + str(i) + '.png'))
                plt.title('query_matrix' + str(i))
                plt.show()
                plt.close(fig)
                fig, ax = plt.subplots()
                ax = ax.matshow(key_matrix[i * 50:i * 50 + 50, :50], cmap=plt.cm.Blues)
                plt.savefig(
                    os.path.join(self.experiment_path, self._get_stage_id(stage), 'key_matrix' + str(i) + '.png'))
                plt.title('key_matrix' + str(i))
                plt.show()
                plt.close(fig)
                fig, ax = plt.subplots()
                ax = ax.matshow(skip_matrix[i * 50:i * 50 + 50, :50], cmap=plt.cm.Blues)
                plt.savefig(
                    os.path.join(self.experiment_path, self._get_stage_id(stage), 'skip_matrix' + str(i) + '.png'))
                plt.title('skip_matrix' + str(i))
                plt.show()
                plt.close(fig)
                '''
                fig, ax = plt.subplots()
                ax = ax.matshow(average_gram, cmap=plt.cm.Blues)
                plt.savefig(os.path.join(self.experiment_path, self._get_stage_id(stage), 'average_gram_matrix.png'))
                plt.title('avg_gram_matrix_')
                plt.show()
                plt.close(fig)
                '''
        return query_matrix, key_matrix, value_matrix, skip_matrix, query_bias, key_bias, value_bias, skip_bias

    def visualize_the_score_between_two_images(self, index_1, index_2, stage):
        the_model = self._get_model(stage=stage)
        the_model.eval()
        raw_data = self._get_raw_data()
        result = np.zeros(4)
        training_params = self._get_training_params()
        assert training_params.optimization_definition_set.model_definition.type == ModelType.GeneralNet, \
            "I can work just with MultiClassGeneralNet"
        query_matrix = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_query.weight.cpu().detach().numpy()
        key_matrix = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_key.weight.cpu().detach().numpy()
        value_matrix = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_value.weight.cpu().detach().numpy()
        query_bias = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_query.bias.cpu().detach().numpy()
        key_bias = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_key.bias.cpu().detach().numpy()
        value_bias = the_model.model.mpn_net.model.mpn_layers[0].trans.lin_key.bias.cpu().detach().numpy()
        # print(query_matrix)
        # print(key_matrix)
        sample_1_feature_vect = the_model.model.classif_net.model.backbone.forward(
            raw_data.dataset_train[index_1][0].unsqueeze(dim=0))
        sample_2_feature_vect = the_model.model.classif_net.model.backbone.forward(
            raw_data.dataset_train[index_2][0].unsqueeze(dim=0))
        sample_1_feature_vect = the_model.model.mpn_net.model.hybrid_fc(sample_1_feature_vect).cpu().detach().numpy()
        sample_2_feature_vect = the_model.model.mpn_net.model.hybrid_fc(sample_2_feature_vect).cpu().detach().numpy()
        # print(sample_1_feature_vect)
        # print(sample_2_feature_vect)
        sample_1_query_vect = sample_1_feature_vect @ query_matrix.T
        sample_1_query_vect = sample_1_query_vect + query_bias
        sample_2_key_vect = sample_2_feature_vect @ key_matrix.T
        sample_2_key_vect = sample_2_key_vect + key_bias
        # print(sample_1_query_vect)
        # print(sample_2_key_vect)
        for i in range(4):
            result[i] = sample_1_query_vect[0, i * 50:(i + 1) * 50] @ sample_2_key_vect[0, i * 50:(i + 1) * 50].T
        return result

    def visualize_attention_weights_of_a_batch(self, stage, indices, head_idx, use_test_set: bool = False):
        # In rows would be attention weights w.r.t. data in the particular row
        the_model = self._get_model(stage=stage)
        datamodule = self._get_datamodule(stage=stage)
        training_params = self._get_training_params()
        assert training_params.optimization_definition_set.model_definition.type == ModelType.GeneralNet, \
            "I can work just with MultiClassGeneralNet"

        batch_size = len(indices)

        if not use_test_set:
            unlabeled_loader = DataLoader(
                datamodule.unaugmented_dataset, batch_size=batch_size,
                sampler=SubsetSequentialSampler(indices),
                num_workers=0,
                drop_last=True,
                # more convenient if we maintain the order of subset
                pin_memory=True)
        if use_test_set:
            unlabeled_loader = DataLoader(
                datamodule.datamodule.dataset_test, batch_size=batch_size,
                sampler=SubsetSequentialSampler(indices),
                num_workers=0,
                drop_last=True,
                # more convenient if we maintain the order of subset
                pin_memory=True)

        layer_to_inspect = "model.mpn_net.model.mpn_layers.layer_0.trans.alpha_dropout"
        handle = the_model.inspect_layer_output(layer_to_inspect)
        # the_model.model.mpn_net.model.mpn_layers.layer_0.trans.alpha_dropout.register_forward_hook(get_output("alpha"))

        for (batch, labels) in tqdm(unlabeled_loader, leave=False, total=len(unlabeled_loader)):
            batch = batch.to(the_model.device)
            labels = labels.to(the_model.device)
            batch_output = the_model(batch)

        handle.remove()
        alpha = the_model.inspected_variables[layer_to_inspect]

        #fig = make_subplots(rows=2, cols=2, subplot_titles=("alpha_0", "alpha_1", "alpha_2", "alpha_3"))

        for batch_alpha in alpha:
            batch_alpha = batch_alpha.view(batch_size, batch_size, 4).T.cpu().detach().numpy()
            for head in range(4):
                if head == head_idx:
                    attention_weights = batch_alpha[head]
                    influences = attention_weights.sum(axis=0).squeeze()
                    #influential = indices[np.argwhere(influences > 10)].squeeze().tolist()
                    most_influential = np.argmax(attention_weights, axis=1)
                    #'''
                    fig_2, ax = plt.subplots()
                    ax = ax.matshow(attention_weights, cmap=plt.cm.Blues)
                    #plt.savefig(
                    #    os.path.join(self.experiment_path, self._get_stage_id(stage), 'TEST_attention_matrix, batch_0_100, head_' + str(head_idx)))
                    #plt.close()
                    #'''
                    plt.show()
                    #fig.add_trace(go.Heatmap(z=attention_weights, colorscale='Blues'),
                    #              row=head // 2 + 1, col=head % 2 + 1)

        #fig.update_traces(showscale=False)
        #fig.show()
        return most_influential


class ImageVisualizer:

    def __init__(self):
        self.raw_data = CIFARDataModule(
            CIFARHyperParameterSet(
                dataset_to_use=DataModuleType.CIFAR10,
                num_workers=0,
                batch_size=100,
                val_ratio=0.0,
                duplication_factor=1,
                train_transforms_def=ToTensorTransformDefinition(),
                val_transforms_def=None,
                test_transforms_def=ToTensorTransformDefinition()))

    def visualise_image(self, index):
        plt.imshow(self.raw_data.dataset_train[index][0].permute(1, 2, 0))
        plt.show()
        print(self.raw_data.dataset_train[index][1])

    def classify_image(self, index):
        return self.raw_data.dataset_train[index][1]
