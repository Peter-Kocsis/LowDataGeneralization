import os

import torch

from lowdataregime.classification.model.graph._graph.edge_attrib import NoEdgeAttributeDefinitionSet
from lowdataregime.classification.model.graph._graph.graph_builder import DenseGraphBuilder, GraphBuilderHyperParameterSet, \
    DistinctGraphBuilder
from lowdataregime.classification.trainer.trainers import PL_TrainerDefinition, PL_TrainerHyperParameterSet, DeviceType
from lowdataregime.parameters.active_loader import ActiveLoaderDefinition, ActiveLoaderHyperParameterSet
from lowdataregime.visualization.active_visualiser import ActiveVisualizer, ActiveVisualizerHyperParameterSet, ImageVisualizer
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    '''
    training_path = 'logs/transformer_effect/cifar10/MultiClassGeneralNet/training_viz_fs50_dist_406116_20210518-092937'

    np.set_printoptions(threshold=sys.maxsize)
    active_visualizer = ActiveVisualizer(ActiveVisualizerHyperParameterSet(
        experiment_path=training_path,
        active_loader_def=ActiveLoaderDefinition(ActiveLoaderHyperParameterSet(
                                experiment_path=training_path)),
        trainer_definition=PL_TrainerDefinition(PL_TrainerHyperParameterSet(runtime_mode=DeviceType.CPU))))


    indices = np.linspace(start=0, stop=99, num=100, dtype=int)

    active_visualizer.visualize_layers_tsne(stage=1,
                                            indices=indices,
                                            layers_to_visualize=["model.classif_net.model.backbone",
                                                                 "model.mpn_net.model.mpn_layers.layer_0"])


    training_path = 'logs/transformer_effect/cifar10/ResNet18/training_viz_resnet_406349_20210518-101828'

    np.set_printoptions(threshold=sys.maxsize)
    active_visualizer = ActiveVisualizer(ActiveVisualizerHyperParameterSet(
        experiment_path=training_path,
        active_loader_def=ActiveLoaderDefinition(ActiveLoaderHyperParameterSet(
            experiment_path=training_path)),
        trainer_definition=PL_TrainerDefinition(PL_TrainerHyperParameterSet(runtime_mode=DeviceType.CPU))))

    indices = np.linspace(start=0, stop=99, num=100, dtype=int)

    active_visualizer.visualize_layers_tsne(stage=1, indices=indices,
                                            layers_to_visualize=["flatten4"])
    '''

    np.set_printoptions(threshold=sys.maxsize)
    pd.options.display.max_rows = 2000
    pd.options.display.max_columns = 100
    active_visualizer = ActiveVisualizer(ActiveVisualizerHyperParameterSet(
        experiment_path='logs/farewell_runs/cifar10/GeneralNet/training_classical_run_509923_20210706-155319',
        active_loader_def=ActiveLoaderDefinition(ActiveLoaderHyperParameterSet(
                                experiment_path='logs/farewell_runs/cifar10/GeneralNet/training_classical_run_509923_20210706-155319')),
        trainer_definition=PL_TrainerDefinition(PL_TrainerHyperParameterSet(runtime_mode=DeviceType.CPU))))

    '''
    active_visualizer.print_easter_egg()
    active_visualizer.evaluate_test_accuracy(stage=9)
    predictions, batches = active_visualizer.obtain_predictions_batches_and_labels_on_labeled_pool(stage=9, memorize=True)
    print(predictions)
    print(batches)
    print(predictions.shape)
    uncertainties = active_visualizer.obtain_uncertainties_on_labeled_pool(stage=9, memorize=True, plot_hist=True)
    print(uncertainties)
    predictions_2, batches_2 = active_visualizer.obtain_predictions_batches_and_labels_on_unlabeled_pool(stage=9, memorize=True)
    print(predictions_2)
    print(batches_2)
    uncertainties_2 = active_visualizer.obtain_uncertainties_on_unlabeled_pool(stage=9, memorize=True, plot_hist=True)
    print(uncertainties_2)
    '''

    #active_visualizer.show_the_attention_matrices(stage=9, draw_matrices=True)
    #imviz = ImageVisualizer()
    #imviz.visualise_image(index=34)
    '''
    for sample in range(100):
        file_to_write = open(f"logs/active_learning/cifar10/MultiClassGeneralNet/training_visuallization_358009_20210415-040203/tryout_{sample}.txt", 'a')
        for i in range(9):
            sample_to_inject = sample
            file_to_write.write("How does {} sample influence the uncertainties on first 20 samples if we increase"
                                "the occurence of it in the batch? First reported uncertainty is for sample {} \n \n".format(sample_to_inject, sample_to_inject))
            indices = np.linspace(start=1, stop=99-10*i, num=99-10*i, dtype=int)
            indices_2 = np.zeros(1+10*i, dtype=int)+sample_to_inject
            indices = np.concatenate((indices_2, indices))
            #indices = np.linspace(start=0, stop=99, num=100, dtype=int)
            file_to_write.write(str(indices))
            file_to_write.write("\n")
            predictions, uncertainties = active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=indices)    
            file_to_write.write(str(uncertainties[0, 10*i:10*i+20].cpu().numpy().round(4)))
            file_to_write.write("\n \n")
        file_to_write.close()
    '''
    '''
    for i in range(9):
        indices = np.linspace(start=1, stop=99 - 10 * i, num=99 - 10 * i, dtype=int)
        indices_2 = np.zeros(1 + 10 * i, dtype=int) + 34
        indices = np.concatenate((indices_2, indices))
        predictions, uncertainties = active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=indices)
        print(predictions[11+10*i])
        print(uncertainties[0, 11+10*i])
    '''
    '''
    counter = 0
    index = 0
    batch = []
    while counter < 99:
        if imviz.classify_image(index) == 7:
            batch.append(index)
            counter += 1
        index += 1
    batch = np.array(batch)
    batch = np.concatenate((np.array([11]), batch))
    predictions, uncertainties = active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=batch)
    print(predictions)
    print(uncertainties)
    '''
    '''
    x = np.linspace(start=0, stop=19, num=20, dtype=int)
    res_1 = np.zeros(20)
    res_2 = np.zeros(20)
    for batch_idx in range(20):
        batch = np.linspace(start=100*batch_idx, stop=100*(batch_idx+1)-1, num=99, dtype=int)
        sample_1 = np.array([1])
        sample_2 = np.array([0])
        batch_1 = np.concatenate((sample_1, batch))
        batch_2 = np.concatenate((sample_2, batch))
        predictions_1, uncertainties_1 = active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=batch_1)
        predictions_2, uncertainties_2 = active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=batch_2)
        res_1[batch_idx] = uncertainties_1[0, 0]
        res_2[batch_idx] = uncertainties_2[0, 0]
    plt.plot(x, res_1, label=f"sample {sample_1}")
    plt.plot(x, res_2, label=f"sample {sample_2}")
    plt.legend()
    plt.xlabel("Batch index")
    plt.ylabel("Uncertainty")
    plt.show()
    '''
    '''
    x = np.linspace(start=0, stop=49, num=50, dtype=int)
    res = np.zeros(50)
    for i in range(50):
        batch = np.zeros(99, dtype=int)+i
        sample = np.array([532], dtype=int)
        batch = np.concatenate((sample, batch))
        predictions, uncertainties = active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=batch)
        res[i] = uncertainties[0, 0]
    plt.plot(x, res, label=f"sample {sample}")
    plt.legend()
    plt.xlabel(f"Duplicate number")
    plt.ylabel(f"Uncertainty")
    plt.title(f"Uncertainty of sample {sample}")
    plt.show()
    '''
    '''
    x = np.linspace(start=0, stop=49, num=50, dtype=int)
    counter = 0
    sample = np.array([0], dtype=int)
    cls = imviz.classify_image(sample.squeeze())
    index = 0
    res = np.zeros(50)
    while counter < 50:
        if imviz.classify_image(index) == cls:
            batch = np.zeros(99, dtype=int)+index
            batch = np.concatenate((sample, batch))
            predictions, uncertainties = active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=batch)
            print(predictions[0])
            res[counter] = uncertainties[0, 0]
            counter += 1
        index += 1
    plt.plot(x, res, label=f"sample {sample}")
    plt.legend()
    plt.xlabel(f"Duplicate")
    plt.ylabel(f"Uncertainty")
    plt.title(f"Uncertainty of sample {sample}, using the same class")
    plt.show()
    '''
    #res = active_visualizer.visualize_the_score_between_two_images(index_1=20, index_2=71, stage=9)*10**41
    #print(res)
    '''
    for j in range(4):
        x = np.linspace(start=0, stop=9, num=10, dtype=int)
        res = np.zeros(10)
        for i in range(10):
            query_mat, key_mat, value_mat, skip_mat, query_bias, key_bias, value_bias, skip_bias = \
                active_visualizer.show_the_attention_matrices(stage=i)
            #print(query_mat.shape)
            #res[i] = np.log10(np.max(np.abs(query_mat)))
            res[i] = np.log10(np.linalg.norm(skip_bias[0+50*j:50+50*j], ord=2))
        plt.plot(x, res)
        plt.xlabel("stage")
        plt.ylabel("log10 l2-norm")
        plt.title("Log10 skip bias l2-norm w.r.t. stage, head: " + str(j))
        plt.savefig("logs/normal_fs50\cifar10\GeneralNet/training_dense_442281_20210603-164350/skip_bias_norm_head " + str(j))
        plt.show()
    '''
    '''
    query_mat, key_mat, value_mat, _, _, _ = active_visualizer.show_the_attention_matrices(stage=0, draw_matrices=True)
    '''
    '''
    indices = np.linspace(start=0, stop=99, num=100, dtype=int)
    result = active_visualizer.visualize_attention_weights_of_a_batch(stage=0, indices=indices)
    print(result)
    '''
    '''
    res = np.zeros((3, 30))
    file_to_write = open("logs/active_learning/cifar10/MultiClassGeneralNet/training_visuallization_358009_20210415-040203/uncertainty_analysis_big.txt", 'w')
    file_to_write.write("Sample\tMean uncertainty\tAbsolute variance\tRelative variance\n")
    for i in range(3):
        print(i)
        for j in range(30):
            print(j)
            batch = np.linspace(start=j*100, stop=j*100+99, num=99, dtype=int)
            sample = np.array([i], dtype=int)
            batch = np.concatenate((sample, batch))
            predictions, uncertainties = active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=batch)
            res[i, j] = uncertainties[0, 0]
        mean = np.mean(res[i, :])
        abs_var = np.var(res[i, :])
        rel_var = np.var(res[i, :]/mean)
        file_to_write.write(f"{i}\t{mean:.4f}\t{abs_var:.3f}\t{rel_var:.3f}\n")
    file_to_write.close()
    '''
    '''
    indices = np.linspace(start=0, stop=99, num=100, dtype=int)
    predictions, uncertainties, iid_predictions, iid_uncertainties = active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=indices, return_backbone_predictions_and_uncertainties=True)
    print(predictions)
    print(iid_predictions)
    print(uncertainties)
    print(iid_uncertainties)
    '''
    '''
    influ = []
    for i in range(5):
        indices = np.linspace(start=0+100*i, stop=99+100*i, num=100, dtype=int)
        #query_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=1, indices=indices, layers_to_inspect=["model.mpn_net.model.mpn_layers.layer_0.trans.lin_query"])
        #query_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=0, indices=indices, layers_to_inspect=["model.mpn_net.model.hybrid_fc"])
        #query_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=0, indices=indices, layers_to_inspect=["model.classif_net.model.backbone"])
        #print(query_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_query"].cpu().detach().numpy()[0, 10, :])
        #print(np.abs(query_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_query"].cpu().detach().numpy()).mean())
        #print(query_vects["model.mpn_net.model.hybrid_fc"].cpu().detach().numpy()[0, 10, :])
        #print(query_vects["model.classif_net.model.backbone"].cpu().detach().numpy().shape)
        #print(query_vects["model.classif_net.model.backbone"].cpu().detach().numpy()[0, 0, 0:30])
        #key_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=1, indices=indices, layers_to_inspect=["model.mpn_net.model.mpn_layers.layer_0.trans.lin_key"])
        #key_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=0, indices=indices, layers_to_inspect=["model.mpn_net.model.hybrid_fc"])
        #key_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=0, indices=indices, layers_to_inspect=["model.classif_net.model.backbone"])
        #print(key_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_key"].cpu().detach().numpy()[0, 0, :])
        #print(np.abs(key_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_key"].cpu().detach().numpy()).mean())
        #print(key_vects["model.mpn_net.model.hybrid_fc"].cpu().detach().numpy()[0, 0, :])
        #print(key_vects["model.classif_net.model.backbone"].cpu().detach().numpy()[0, 0, :])
        res = active_visualizer.visualize_attention_weights_of_a_batch(stage=0, indices=indices)
        influ.append(res)
    print(influ)
    '''
    '''
    indices = np.linspace(start=0, stop=49, num=50, dtype=int)
    #indices = np.concatenate((np.array([209]), indices), axis=0)
    res = active_visualizer.visualize_attention_weights_of_a_batch(stage=0, indices=indices)
    print(res)
    '''
    '''
    i = 0
    for j in range(10):
        print(j)
        for k in range(4):
            print(k)
            indices = np.linspace(start=0+200*i, stop=199+200*i, num=200, dtype=int)
            feature_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=j, indices=indices, layers_to_inspect=["model.mpn_net.model.mpn_layers.layer_0.trans.lin_key"],
                                                                               use_test_dataset=True)
            feature_vects = feature_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_key"].cpu().detach().numpy()
            feature_vects = feature_vects.squeeze()
            print(feature_vects.shape)
            feature_vects = feature_vects[:, (0+50*k):(50+50*k)]
            helping = np.linspace(start=0, stop=39999, num=40000, dtype=int)
            helping = helping % 200 == 0
            feature_vects = feature_vects[helping, :]
            _, _, _, _, query_bias, _, _, _ = active_visualizer.show_the_attention_matrices(stage=j)
            query_bias = query_bias[(0+50*k):(50+50*k)]
            query_bias = np.expand_dims(query_bias, axis=1)
            partial_scores = feature_vects @ query_bias
            partial_scores = partial_scores.squeeze()
            #feature_sizes = np.abs(feature_vects).max(axis=1)
            influences = active_visualizer.visualize_attention_weights_of_a_batch(stage=j, indices=indices, head_idx=k,
                                                                                  use_test_set=True).squeeze()
            plt.scatter(partial_scores, np.log(influences))
            plt.title("Dependance between the influenciality and influence on query bias, \n indices: " + str(0+200*i) +
                      ":" + str(199+200*i))
            plt.xlabel("dot product keys * queries' bias")
            plt.ylabel("sum of influences")
            file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=j), 'TEST_batch_' + str(i) + '; head_' + str(k))
            plt.savefig(file_path)
            plt.close()
            feature_vects = {}
    '''
    '''
    i = 0
    for j in range(10):
        print(j)
        for k in range(4):
            print(k)
            indices = np.linspace(start=0 + 200 * i, stop=199 + 200 * i, num=200, dtype=int)
            feature_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=j, indices=indices, layers_to_inspect=[
                "model.mpn_net.model.mpn_layers.layer_0.trans.lin_key"], use_test_dataset=True)
            feature_vects = feature_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_key"].cpu().detach().numpy()
            feature_vects = feature_vects.squeeze()
            feature_vects = feature_vects[:, (0+50*k):(50+50*k)]
            helping = np.linspace(start=0, stop=39999, num=40000, dtype=int)
            helping = helping % 200 == 0
            feature_vects = feature_vects[helping, :]
            feature_sizes = np.sqrt((feature_vects ** 2).sum(axis=1))
            influences = active_visualizer.visualize_attention_weights_of_a_batch(stage=j, indices=indices, head_idx=k, use_test_set=True).squeeze()
            plt.scatter(feature_sizes, np.log(influences))
            plt.title("Dependance between the influenciality and the size of key vector,\n indices: " + str(0 + 200 * i) +
                      ":" + str(199 + 200 * i))
            plt.xlabel("key vector size")
            plt.ylabel("log sum of influences")
            file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=j), 'TEST_sizes_batch_' + str(i) + '; head_' + str(k))
            plt.savefig(file_path)
            plt.close()
    '''
    '''
    indices = np.linspace(start=0, stop=99, num=100, dtype=int)
    backbone_features, iid_scores, iid_predictions, iid_uncertainties, iid_features, mpn_features, mpn_scores, \
    mpn_predictions, mpn_uncertainties, targets = \
        active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=indices)
    print(backbone_features.shape)
    print(iid_scores.shape)
    print(iid_predictions.shape)
    print(iid_uncertainties.shape)
    print(iid_features.shape)
    print(mpn_features.shape)
    print(mpn_scores.shape)
    print(mpn_predictions.shape)
    print(mpn_uncertainties.shape)
    print(targets.shape)
    '''


    #'''
    batch_size = 100
    for batch_idx in range(0, 2):
        print(batch_idx)
        batch_indices = np.linspace(start=0+batch_size*batch_idx, stop=batch_size-1+batch_size*batch_idx, num=batch_size, dtype=int)
        for stage_idx in range(0, 10):
            print(stage_idx)
            backbone_features, iid_scores, iid_predictions, iid_uncertainties, iid_features, mpn_features, mpn_scores, \
            mpn_predictions, mpn_uncertainties, targets = \
                active_visualizer.evaluate_model_on_custom_batch(stage=stage_idx, indices=batch_indices, use_test_set=True)
            backbone_features = backbone_features.cpu().detach().numpy()
            iid_scores = iid_scores.cpu().detach().numpy()
            iid_predictions = iid_predictions.cpu().detach().numpy()
            iid_final_predictions = np.argmax(iid_predictions, axis=1).squeeze()
            iid_uncertainties = iid_uncertainties.cpu().detach().numpy().squeeze()
            #iid_features = iid_features.cpu().detach().numpy()
            mpn_features = mpn_features.cpu().detach().numpy()
            mpn_scores = mpn_scores.cpu().detach().numpy()
            mpn_predictions = mpn_predictions.cpu().detach().numpy()
            mpn_final_predictions = np.argmax(mpn_predictions, axis=1).squeeze()
            mpn_uncertainties = mpn_uncertainties.cpu().detach().numpy().squeeze()
            targets = targets.cpu().detach().numpy()
            backbone_features_norm = np.linalg.norm(backbone_features, ord=2, axis=1)
            iid_scores_norm = np.linalg.norm(iid_scores, ord=2, axis=1)
            mpn_features_norm = np.linalg.norm(mpn_features, ord=2, axis=1)
            mpn_scores_norm = np.linalg.norm(mpn_scores, ord=2, axis=1)

            _, _, _, _, _, self_loop_mpn_features, self_loop_mpn_scores, \
            self_loop_mpn_predictions, self_loop_mpn_uncertainties, _ = \
                active_visualizer.evaluate_model_on_custom_batch(stage=stage_idx, distinct_graph=True,
                                                                 indices=batch_indices, use_test_set=True)

            self_loop_mpn_features = self_loop_mpn_features.cpu().detach().numpy()
            self_loop_mpn_scores = self_loop_mpn_scores.cpu().detach().numpy()
            self_loop_mpn_predictions = self_loop_mpn_predictions.cpu().detach().numpy()
            self_loop_mpn_uncertainties = self_loop_mpn_uncertainties.cpu().detach().numpy().squeeze()
            self_loop_mpn_final_predictions = np.argmax(self_loop_mpn_predictions, axis=1).squeeze()
            self_loop_mpn_features_norm = np.linalg.norm(self_loop_mpn_features, ord=2, axis=1)
            self_loop_mpn_scores_norm = np.linalg.norm(self_loop_mpn_scores, ord=2, axis=1)

            _, _, _, _, _, just_skip_mpn_features, just_skip_mpn_scores, \
            just_skip_mpn_predictions, just_skip_mpn_uncertainties, _ = \
                active_visualizer.evaluate_model_on_custom_batch(stage=stage_idx, indices=batch_indices,
                                                                 use_test_set=True, use_just_skip_layer=True)

            just_skip_mpn_features = just_skip_mpn_features.cpu().detach().numpy()
            just_skip_mpn_scores = just_skip_mpn_scores.cpu().detach().numpy()
            just_skip_mpn_predictions = just_skip_mpn_predictions.cpu().detach().numpy()
            just_skip_mpn_uncertainties = just_skip_mpn_uncertainties.cpu().detach().numpy().squeeze()
            just_skip_mpn_final_predictions = np.argmax(just_skip_mpn_predictions, axis=1).squeeze()
            just_skip_mpn_features_norm = np.linalg.norm(just_skip_mpn_features, ord=2, axis=1)
            just_skip_mpn_scores_norm = np.linalg.norm(just_skip_mpn_scores, ord=2, axis=1)

            graph_builder = DenseGraphBuilder(GraphBuilderHyperParameterSet(NoEdgeAttributeDefinitionSet()))
            _, edge_index = graph_builder.get_graph(iid_features)
            after_mpn_features = active_visualizer._get_model(stage=stage_idx).eval().model.mpn_net.model.mpn_layers.layer_0.trans.forward(iid_features, edge_index).cpu().detach().numpy()
            after_mpn_features_norm = np.linalg.norm(after_mpn_features, ord=2, axis=1)

            self_loop_graph_builder = DistinctGraphBuilder(GraphBuilderHyperParameterSet(NoEdgeAttributeDefinitionSet()))
            _, self_loop_edge_index = self_loop_graph_builder.get_graph(iid_features)
            self_loop_after_mpn_features = active_visualizer._get_model(stage=stage_idx).eval().model.mpn_net.model.mpn_layers.layer_0.trans.forward(iid_features, self_loop_edge_index).cpu().detach().numpy()
            self_loop_after_mpn_features_norm = np.linalg.norm(self_loop_after_mpn_features, ord=2, axis=1)

            iid_features = iid_features.cpu().detach().numpy()
            iid_features_norm = np.linalg.norm(iid_features, ord=2, axis=1)

            query_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=stage_idx, indices=batch_indices,
                                                                             layers_to_inspect=["model.mpn_net.model.mpn_layers.layer_0.trans.lin_query"], use_test_dataset=True)
            query_vects = query_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_query"].cpu().detach().numpy()
            query_vects = query_vects.squeeze()
            query_vects = query_vects[0:batch_size]
            query_vects = query_vects.reshape((-1, 4, 50))

            key_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=stage_idx, indices=batch_indices, layers_to_inspect=[
                "model.mpn_net.model.mpn_layers.layer_0.trans.lin_key"], use_test_dataset=True)
            key_vects = key_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_key"].cpu().detach().numpy()
            key_vects = key_vects.squeeze()
            helping = np.linspace(start=0, stop=batch_size**2-1, num=batch_size**2, dtype=int)
            helping = helping % batch_size == 0
            key_vects = key_vects[helping, :]
            key_vects = key_vects.reshape((-1, 4, 50))

            value_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=stage_idx, indices=batch_indices,
                                                                           layers_to_inspect=[
                                                                               "model.mpn_net.model.mpn_layers.layer_0.trans.lin_value"], use_test_dataset=True)
            value_vects = value_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_value"].cpu().detach().numpy()
            value_vects = value_vects.squeeze()
            value_vects = value_vects[helping, :]
            value_vects = value_vects.reshape((-1, 4, 50))

            skip_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=stage_idx, indices=batch_indices,
                                                                             layers_to_inspect=[
                                                                                 "model.mpn_net.model.mpn_layers.layer_0.trans.lin_skip"], use_test_dataset=True)
            skip_vects = skip_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_skip"].cpu().detach().numpy()
            skip_vects = skip_vects.squeeze()

            final_value_vects = after_mpn_features - skip_vects
            final_value_vects_norm = np.linalg.norm(final_value_vects, ord=2, axis=1)

            self_loop_final_value_vects = self_loop_after_mpn_features - skip_vects
            self_loop_final_value_vects_norm = np.linalg.norm(self_loop_final_value_vects, ord=2, axis=1)

            query_vects_norm = np.linalg.norm(query_vects, ord=2, axis=2)
            key_vects_norm = np.linalg.norm(key_vects, ord=2, axis=2)
            value_vects_norm = np.linalg.norm(value_vects, ord=2, axis=2)
            skip_vects_norm = np.linalg.norm(skip_vects, ord=2, axis=1)

            query_matrix, key_matrix, value_matrix, skip_matrix, query_bias, key_bias, value_bias, skip_bias = \
                active_visualizer.show_the_attention_matrices(stage=stage_idx)
            query_matrix = query_matrix.T.reshape((-1, 4, 50))
            key_matrix = key_matrix.T.reshape((-1, 4, 50))
            value_matrix = value_matrix.T.reshape((-1, 4, 50))

            query_bias_norm = np.linalg.norm(query_bias, ord=2)
            query_bias = query_bias.reshape((4, 50))

            query_bias_key_vects_dot_prods = np.zeros((100, 4))
            sum_influences = np.zeros((100, 4))
            query_matrix_norms = np.zeros(4)
            key_matrix_norms = np.zeros(4)
            value_matrix_norms = np.zeros(4)
            for head_idx in range(4):
                query_bias_key_vects_dot_prods[:, head_idx] = key_vects[:, head_idx, :] @ query_bias[head_idx, :].T
                query_matrix_norms[head_idx] = np.linalg.norm(query_matrix[:, head_idx, :], ord=2)
                key_matrix_norms[head_idx] = np.linalg.norm(key_matrix[:, head_idx, :], ord=2)
                value_matrix_norms[head_idx] = np.linalg.norm(value_matrix[:, head_idx, :], ord=2)
                sum_influences[:, head_idx] = active_visualizer.visualize_attention_weights_of_a_batch(stage=stage_idx, indices=batch_indices, head_idx=head_idx, use_test_set=True)

            skip_matrix_norm = np.linalg.norm(skip_matrix, ord=2)
            key_bias_norm = np.linalg.norm(key_bias, ord=2)
            value_bias_norm = np.linalg.norm(value_bias, ord=2)
            skip_bias_norm = np.linalg.norm(skip_bias, ord=2)



            file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                     'TEST_big_analysis_batch_' + str(0+batch_size*batch_idx) + '_to_' +
                                     str(batch_size-1+batch_size*batch_idx) + '.txt')
            file_to_write = open(file_path, 'w')
            print("sample_num\tclass\tiid_class_pred\tmpn_class_pred\tself_loop_mpn_class_pred\tjust_skip_mpn_class_pred\t"
                  "backbone_features_norm\t"
                  "iid_scores_norm\tiid_uncertainty\tfeatures_before_mpn_norm\tfeatures_after_mpn_norm\t"
                  "mpn_features_norm\tmpn_scores_norm\tmpn_uncertainty\tself_loops_after_mpn_features_norm\t"
                  "self_loops_mpn_features_norm\tself_loops_mpn_scores_norm\tself_loops_uncertainty\t"
                  "just_skip_mpn_features_norm\tjust_skip_mpn_scores_norm\tjust_skip_mpn_uncertainty\t"
                  "key_vects_head_0_norm\tkey_vects_head_1_norm\t"
                  "key_vects_head_2_norm\tkey_vects_head_3_norm\tquery_vects_head_0_norm\tquery_vects_head_1_norm\t"
                  "query_vects_head_2_norm\tquery_vects_head_3_norm\tvalue_vects_head_0_norm\tvalue_vects_head_1_norm\t"
                  "value_vects_head_2_norm\tvalue_vects_head_3_norm\tskip_vects_norm\tfinal_value_vects_norm\t"
                  "self_loops_final_value_vects_norm\t"
                  "query_bias_key_vects_dot_prods_head_0\tquery_bias_key_vects_dot_prods_head_1\t"
                  "query_bias_key_vects_dot_prods_head_2\tquery_bias_key_vects_dot_prods_head_3\t"
                  "query_matrix_head_0_norm\tquery_matrix_head_1_norm\tquery_matrix_head_2_norm\t"
                  "query_matrix_head_3_norm\tkey_matrix_head_0_norm\tkey_matrix_head_1_norm\t"
                  "key_matrix_head_2_norm\tkey_matrix_head_3_norm\tvalue_matrix_head_0_norm\t"
                  "value_matrix_head_1_norm\tvalue_matrix_head_2_norm\tkey_matrix_head_3_norm\t"
                  "skip_matrix_norm\tquery_bias_norm\tkey_bias_norm\tvalue_bias_norm\t"
                  "skip_bias_norm\tinfluentiality_head_0\tinfluentiality_head_1\tinfluentiality_head_2\t"
                  "influentiality_head_3", file=file_to_write, flush=True)
            for sample_idx in range(batch_size):
                print(str(sample_idx+batch_idx*batch_size) + "\t" + str(targets[sample_idx]) + "\t" +
                      str(iid_final_predictions[sample_idx]) + "\t" + str(mpn_final_predictions[sample_idx]) + "\t" +
                      str(self_loop_mpn_final_predictions[sample_idx]) + "\t" + str(just_skip_mpn_final_predictions[sample_idx]) + "\t" +
                      str(backbone_features_norm[sample_idx]) + "\t" + str(iid_scores_norm[sample_idx]) + "\t" +
                      str(iid_uncertainties[sample_idx]) + "\t" + str(iid_features_norm[sample_idx]) + "\t" +
                      str(after_mpn_features_norm[sample_idx]) + "\t" + str(mpn_features_norm[sample_idx]) + "\t" +
                      str(mpn_scores_norm[sample_idx]) + "\t" + str(mpn_uncertainties[sample_idx]) + "\t" +
                      str(self_loop_after_mpn_features_norm[sample_idx]) + "\t" + str(self_loop_mpn_features_norm[sample_idx]) + "\t" +
                      str(self_loop_mpn_scores_norm[sample_idx]) + "\t" + str(self_loop_mpn_uncertainties[sample_idx]) + "\t" +
                      str(just_skip_mpn_features_norm[sample_idx]) + "\t" + str(just_skip_mpn_scores_norm[sample_idx]) + "\t" +
                      str(just_skip_mpn_uncertainties[sample_idx]) + "\t" +
                      str(key_vects_norm[sample_idx, 0]) + "\t" + str(key_vects_norm[sample_idx, 1]) + "\t" +
                      str(key_vects_norm[sample_idx, 2]) + "\t" + str(key_vects_norm[sample_idx, 3]) + "\t" +
                      str(query_vects_norm[sample_idx, 0]) + "\t" + str(query_vects_norm[sample_idx, 1]) + "\t" +
                      str(query_vects_norm[sample_idx, 2]) + "\t" + str(query_vects_norm[sample_idx, 3]) + "\t" +
                      str(value_vects_norm[sample_idx, 0]) + "\t" + str(value_vects_norm[sample_idx, 1]) + "\t" +
                      str(value_vects_norm[sample_idx, 2]) + "\t" + str(value_vects_norm[sample_idx, 3]) + "\t" +
                      str(skip_vects_norm[sample_idx]) + "\t" + str(final_value_vects_norm[sample_idx]) + "\t" +
                      str(self_loop_final_value_vects_norm[sample_idx]) + "\t" +
                      str(query_bias_key_vects_dot_prods[sample_idx, 0]) + "\t" + str(query_bias_key_vects_dot_prods[sample_idx, 1]) + "\t" +
                      str(query_bias_key_vects_dot_prods[sample_idx, 2]) + "\t" + str(query_bias_key_vects_dot_prods[sample_idx, 3]) + "\t" +
                      str(query_matrix_norms[0]) + "\t" + str(query_matrix_norms[1]) + "\t" +
                      str(query_matrix_norms[2]) + "\t" + str(query_matrix_norms[3]) + "\t" +
                      str(key_matrix_norms[0]) + "\t" + str(key_matrix_norms[1]) + "\t" +
                      str(key_matrix_norms[2]) + "\t" + str(key_matrix_norms[3]) + "\t" +
                      str(value_matrix_norms[0]) + "\t" + str(value_matrix_norms[1]) + "\t" +
                      str(value_matrix_norms[2]) + "\t" + str(value_matrix_norms[3]) + "\t" +
                      str(skip_matrix_norm) + "\t" + str(query_bias_norm) + "\t" +
                      str(key_bias_norm) + "\t" + str(value_bias_norm) + "\t" +
                      str(skip_bias_norm) + "\t" + str(sum_influences[sample_idx, 0]) + "\t" +
                      str(sum_influences[sample_idx, 1]) + "\t" + str(sum_influences[sample_idx, 2]) + "\t" +
                      str(sum_influences[sample_idx, 3]), file=file_to_write, flush=True)
            file_to_write.close()

    #'''
    
    '''
    dt = pd.read_csv("logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_0/big_analysis_batch_0_to_99.txt", delim_whitespace=True)
    unc_diff = np.log10(np.abs(dt["iid_uncertainty"]-dt["mpn_uncertainty"])/dt["iid_uncertainty"])
    norm_ratio = dt["final_value_vects_norm"]/dt["skip_vects_norm"]
    plt.scatter(norm_ratio, unc_diff)
    plt.show()
    '''
    '''
    dts = []
    for stage_idx in range(10):
        dts.append(pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_all_indices_0_to_1999.txt", delim_whitespace=True))
    '''
    '''
    classic_vs_self_loops = np.zeros(10)
    classic_vs_just_skips = np.zeros(10)
    just_skips_vs_self_loops = np.zeros(10)

    for stage_idx in range(10):
        dt_0 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_0_to_99.txt", delim_whitespace=True)
        dt_1 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_100_to_199.txt", delim_whitespace=True)
        dt_2 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_200_to_299.txt", delim_whitespace=True)
        dt_3 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_300_to_399.txt", delim_whitespace=True)
        dt_4 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_400_to_499.txt", delim_whitespace=True)
        dt_5 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_500_to_599.txt", delim_whitespace=True)
        dt_6 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_600_to_699.txt", delim_whitespace=True)
        dt_7 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_700_to_799.txt", delim_whitespace=True)
        dt_8 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_800_to_899.txt", delim_whitespace=True)
        dt_9 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_900_to_999.txt", delim_whitespace=True)
        dt_10 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1000_to_1099.txt", delim_whitespace=True)
        dt_11 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1100_to_1199.txt", delim_whitespace=True)
        dt_12 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1200_to_1299.txt", delim_whitespace=True)
        dt_13 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1300_to_1399.txt", delim_whitespace=True)
        dt_14 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1400_to_1499.txt", delim_whitespace=True)
        dt_15 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1500_to_1599.txt", delim_whitespace=True)
        dt_16 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1600_to_1699.txt", delim_whitespace=True)
        dt_17 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1700_to_1799.txt", delim_whitespace=True)
        dt_18 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1800_to_1899.txt", delim_whitespace=True)
        dt_19 = pd.read_csv(f"logs/normal_fs50/cifar10/GeneralNet/training_dense_442281_20210603-164350/stage_{stage_idx}/TEST_big_analysis_batch_1900_to_1999.txt", delim_whitespace=True)
        dt = pd.concat((dt_0, dt_1, dt_2, dt_3, dt_4, dt_5, dt_6, dt_7, dt_8, dt_9, dt_10, dt_11, dt_12, dt_13, dt_14, dt_15, dt_16, dt_17, dt_18, dt_19),  ignore_index=True)
        #print(dt[['mpn_uncertainty', 'self_loops_uncertainty', 'just_skip_mpn_uncertainty']])
        #print(dt[['mpn_uncertainty', 'self_loops_uncertainty', 'just_skip_mpn_uncertainty']].apply(np.argsort).apply(np.argsort))
        temp = dt[['mpn_uncertainty', 'self_loops_uncertainty', 'just_skip_mpn_uncertainty']].apply(np.argsort)[1800:1999].apply(np.sort)
        classic_vs_self_loops[stage_idx] = np.intersect1d(temp['mpn_uncertainty'], temp['self_loops_uncertainty']).size/200
        classic_vs_just_skips[stage_idx] = np.intersect1d(temp['mpn_uncertainty'], temp['just_skip_mpn_uncertainty']).size / 200
        just_skips_vs_self_loops[stage_idx] = np.intersect1d(temp['just_skip_mpn_uncertainty'], temp['self_loops_uncertainty']).size / 200

    #print(np.intersect1d(temp['mpn_uncertainty'], temp['self_loops_uncertainty']).size/50)
    #print(np.intersect1d(temp['mpn_uncertainty'], temp['just_skip_mpn_uncertainty']).size/50)
    #print(np.intersect1d(temp['self_loops_uncertainty'], temp['just_skip_mpn_uncertainty']).size/50)
    #print()
    plt.plot(classic_vs_self_loops)
    plt.title("classic mpn and self-loops mpn intersection")
    plt.xlabel("stage")
    plt.ylabel("percentage")
    plt.show()
    plt.close()
    plt.plot(classic_vs_just_skips)
    plt.title("classic mpn and just-skip intersection")
    plt.xlabel("stage")
    plt.ylabel("percentage")
    plt.show()
    plt.close()
    plt.plot(just_skips_vs_self_loops)
    plt.title("just-skip and self-loops intersection")
    plt.xlabel("stage")
    plt.ylabel("percentage")
    plt.show()
    plt.close()
    '''
    '''
    iid_vs_classic = np.zeros(10)
    iid_vs_just_skips = np.zeros(10)
    iid_vs_self_loops = np.zeros(10)

    for stage_idx in range(10):
        temp = dts[stage_idx][['iid_uncertainty', 'mpn_uncertainty', 'self_loops_uncertainty', 'just_skip_mpn_uncertainty']].apply(np.argsort)[
               1800:1999].apply(np.sort)
        iid_vs_classic[stage_idx] = np.intersect1d(temp['iid_uncertainty'],
                                                          temp['mpn_uncertainty']).size / 200
        iid_vs_just_skips[stage_idx] = np.intersect1d(temp['iid_uncertainty'],
                                                          temp['self_loops_uncertainty']).size / 200
        iid_vs_self_loops[stage_idx] = np.intersect1d(temp['iid_uncertainty'],
                                                             temp['just_skip_mpn_uncertainty']).size / 200

    plt.plot(iid_vs_classic)
    plt.title("iid and classic mpn intersection")
    plt.xlabel("stage")
    plt.ylabel("percentage")
    plt.show()
    plt.close()
    plt.plot(iid_vs_just_skips)
    plt.title("iid and just-skip intersection")
    plt.xlabel("stage")
    plt.ylabel("percentage")
    plt.show()
    plt.close()
    plt.plot(iid_vs_self_loops)
    plt.title("iid and self-loops intersection")
    plt.xlabel("stage")
    plt.ylabel("percentage")
    plt.show()
    plt.close()
    '''
    '''
    iid_acs = np.zeros(10)
    mpn_acs = np.zeros(10)
    self_loop_acs = np.zeros(10)
    just_skip_acs = np.zeros(10)
    iid_vs_mpn_intersection = np.zeros(10)
    iid_vs_self_loops_intersection = np.zeros(10)
    iid_vs_just_skip_intersection = np.zeros(10)
    mpn_vs_self_loops_intersection = np.zeros(10)
    mpn_vs_just_skip_intersection = np.zeros(10)
    self_loops_vs_just_skip_intersection = np.zeros(10)

    for stage_idx in range(10):
        #print(dts[i][['class', 'iid_class_pred', 'mpn_class_pred', 'self_loop_mpn_class_pred', 'just_skip_mpn_class_pred']])
        iid_acs[stage_idx] = np.sum(dts[stage_idx]['class'] == dts[stage_idx]['iid_class_pred']) / 2000
        mpn_acs[stage_idx] = np.sum(dts[stage_idx]['class'] == dts[stage_idx]['mpn_class_pred']) / 2000
        self_loop_acs[stage_idx] = np.sum(dts[stage_idx]['class'] == dts[stage_idx]['self_loop_mpn_class_pred']) / 2000
        just_skip_acs[stage_idx] = np.sum(dts[stage_idx]['class'] == dts[stage_idx]['just_skip_mpn_class_pred']) / 2000
        iid_vs_mpn_intersection[stage_idx] = np.sum(dts[stage_idx]['iid_class_pred'] == dts[stage_idx]['mpn_class_pred']) / 2000
        iid_vs_self_loops_intersection[stage_idx] = np.sum(dts[stage_idx]['iid_class_pred'] == dts[stage_idx]['self_loop_mpn_class_pred']) / 2000
        iid_vs_just_skip_intersection[stage_idx] = np.sum(dts[stage_idx]['iid_class_pred'] == dts[stage_idx]['just_skip_mpn_class_pred']) / 2000
        mpn_vs_self_loops_intersection[stage_idx] = np.sum(dts[stage_idx]['mpn_class_pred'] == dts[stage_idx]['self_loop_mpn_class_pred']) / 2000
        mpn_vs_just_skip_intersection[stage_idx] = np.sum(dts[stage_idx]['mpn_class_pred'] == dts[stage_idx]['just_skip_mpn_class_pred']) / 2000
        self_loops_vs_just_skip_intersection[stage_idx] = np.sum(dts[stage_idx]['self_loop_mpn_class_pred'] == dts[stage_idx]['just_skip_mpn_class_pred']) / 2000

    plt.plot(iid_acs, label='iid accuracy')
    plt.plot(mpn_acs, label='mpn accuracy')
    plt.plot(self_loop_acs, label='self-loops accuracy')
    plt.plot(just_skip_acs, label='just-skip accuracy')
    plt.title("Accuracies")
    plt.xlabel("stage")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(iid_vs_mpn_intersection, label='iid vs mpn intersection')
    plt.plot(iid_vs_self_loops_intersection, label='iid vs self-loops intersection')
    plt.plot(iid_vs_just_skip_intersection, label='iid vs just-skip intersection')
    plt.plot(mpn_vs_self_loops_intersection, label='mpn vs self-loops intersection')
    plt.plot(mpn_vs_just_skip_intersection, label='mpn vs just-skip intersection')
    plt.plot(self_loops_vs_just_skip_intersection, label='self-loops vs just-skip intersection')
    plt.title("Class prediction intersections")
    plt.xlabel("stage")
    plt.ylabel("percentage")
    plt.legend()
    plt.show()
    plt.close()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['mpn_uncertainty']), np.log(dts[stage_idx]['influentiality_head_0'] +
                                                                      dts[stage_idx]['influentiality_head_1'] +
                                                                      dts[stage_idx]['influentiality_head_2'] +
                                                                      dts[stage_idx]['influentiality_head_3']))
        plt.title(f'Log uncertainty vs. average influentiality, stage_{stage_idx}')
        plt.xlabel('log uncertainty')
        plt.ylabel('mean influentiality')
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['features_before_mpn_norm']), np.log(dts[stage_idx]['features_after_mpn_norm']))
        plt.title(f'Features before and after MPN norms, stage_{stage_idx}')
        plt.xlabel('before mpn norm')
        plt.ylabel('after mpn norm')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'before_vs_after_MPN_features_norm, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['backbone_features_norm']),
                    np.log(dts[stage_idx]['iid_uncertainty']))
        plt.title(f'Backbone features norm vs iid uncertainties, \n stage_{stage_idx}')
        plt.xlabel('backbone features norm')
        plt.ylabel('iid_uncertainty')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'backbone_features_norm_vs_iid_uncertainty, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['backbone_features_norm']),
                    np.log(dts[stage_idx]['mpn_uncertainty']))
        plt.title(f'Backbone features norm vs mpn uncertainties, \n stage_{stage_idx}')
        plt.xlabel('backbone features norm')
        plt.ylabel('mpn_uncertainty')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'backbone_features_norm_vs_mpn_uncertainty, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['features_before_mpn_norm']),
                    np.log(dts[stage_idx]['mpn_uncertainty']))
        plt.title(f'Features before mpn norm vs mpn uncertainties, \n stage_{stage_idx}')
        plt.xlabel('features before mpn norm')
        plt.ylabel('mpn_uncertainty')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'features_before_mpn_norm_vs_mpn_uncertainty, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['features_after_mpn_norm']),
                    np.log(dts[stage_idx]['mpn_uncertainty']))
        plt.title(f'Features after mpn norm vs mpn uncertainties, \n stage_{stage_idx}')
        plt.xlabel('features after mpn norm')
        plt.ylabel('mpn_uncertainty')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'features_after_mpn_norm_vs_mpn_uncertainty, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['mpn_features_norm']),
                    np.log(dts[stage_idx]['mpn_uncertainty']))
        plt.title(f'Features totally after mpn norm vs mpn uncertainties, \n stage_{stage_idx}')
        plt.xlabel('features after mpn norm')
        plt.ylabel('mpn_uncertainty')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'features_totally_after_mpn_norm_vs_mpn_uncertainty, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['mpn_features_norm']),
                    np.log(dts[stage_idx]['mpn_scores_norm']))
        plt.title(f'Features totally after mpn norm vs mpn scores norm, \n stage_{stage_idx}')
        plt.xlabel('features totally after mpn norm')
        plt.ylabel('mpn_scores_norm')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'features_totally_after_mpn_norm_vs_mpn_scores_norm, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['mpn_scores_norm']),
                    np.log(dts[stage_idx]['mpn_uncertainty']))
        plt.title(f'mpn scores norm vs mpn uncertainty, \n stage_{stage_idx}')
        plt.xlabel('mpn scores norm')
        plt.ylabel('mpn uncertainty')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'mpn_scores_norm_vs_mpn_uncertainties, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['iid_scores_norm']),
                    np.log(dts[stage_idx]['iid_uncertainty']))
        plt.title(f'iid scores norm vs iid uncertainty, \n stage_{stage_idx}')
        plt.xlabel('iid scores norm')
        plt.ylabel('iid uncertainty')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'iid_scores_norm_vs_iid_uncertainties, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['features_before_mpn_norm']),
                    np.log(dts[stage_idx]['mpn_features_norm']))
        plt.title(f'iid features norm vs mpn features norm, \n stage_{stage_idx}')
        plt.xlabel('iid features norm')
        plt.ylabel('mpn features norm')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'iid_features_norm_vs_mpn_features_norm, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    backbone_features, iid_scores, iid_predictions, iid_uncertainties, iid_features, \
        mpn_features, mpn_scores, mpn_predictions, mpn_uncertainties, targets = \
        active_visualizer.evaluate_model_on_custom_batch(stage=9, indices=np.linspace(start=0, stop=499, num=500,
                                                         dtype=int), use_test_set=True)

    iid_scores = iid_scores.cpu().detach().numpy()
    iid_uncertainties = iid_uncertainties.cpu().detach().numpy()
    iid_scores_norm = np.linalg.norm(iid_scores, ord=2, axis=1)

    mpn_scores = mpn_scores.cpu().detach().numpy()
    mpn_scores_norm = np.linalg.norm(mpn_scores, ord=2, axis=1)
    mpn_uncertainties = mpn_uncertainties.cpu().detach().numpy()

    plt.scatter(np.log(iid_scores_norm), np.log(iid_uncertainties))
    plt.show()
    plt.close()

    plt.scatter(np.log(mpn_scores_norm), np.log(mpn_uncertainties))
    plt.show()
    plt.close()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['just_skip_mpn_scores_norm']),
                    np.log(dts[stage_idx]['just_skip_mpn_uncertainty']))
        plt.title(f'just skip scores norm vs just skip uncertainty, \n stage_{stage_idx}')
        plt.xlabel('just skip scores norm')
        plt.ylabel('just skip uncertainty')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'just_skip_scores_norm_vs_just_skip_uncertainties, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(np.log(dts[stage_idx]['just_skip_mpn_scores_norm']),
                    np.log(dts[stage_idx]['just_skip_mpn_uncertainty']))
        plt.title(f'just skip scores norm vs just skip uncertainty, \n stage_{stage_idx}')
        plt.xlabel('just skip scores norm')
        plt.ylabel('just skip uncertainty')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'just_skip_scores_norm_vs_just_skip_uncertainties, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.hist(dts[stage_idx]['final_value_vects_norm'] /
                 dts[stage_idx]['skip_vects_norm'], bins=20)
        plt.title(f'Final value vects norms vs skip vects norms, \n stage_{stage_idx}')
        plt.xlabel('stage')
        plt.ylabel('ratio')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'value vects over skip vects norms ratios, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(dts[stage_idx]['iid_uncertainty'], dts[stage_idx]['final_value_vects_norm'] /
                    dts[stage_idx]['skip_vects_norm'])
        plt.title(f'iid uncertainties vs value/skip norms ratio, \n stage_{stage_idx}')
        plt.xlabel('iid_uncertainty')
        plt.ylabel('value/skip norms ratio')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'iid_uncertainties_vs_value_over_skip_norms_ratios, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
        plt.close()
    '''
    '''
    for stage_idx in range(10):
        plt.scatter(dts[stage_idx]['final_value_vects_norm'] / dts[stage_idx]['skip_vects_norm'],
                    dts[stage_idx]['mpn_uncertainty'] / dts[stage_idx]['just_skip_mpn_uncertainty'])
        plt.title(f'relative change of uncertainties vs. value/skip norms ratios, \n stage_{stage_idx}')
        plt.xlabel('value/skip norms ratios')
        plt.ylabel('relative uncartainty change')
        file_path = os.path.join(active_visualizer._get_logger_folder_path(stage=stage_idx),
                                 f'relative_uncertainty_change_vs_value_over_skip_norms_ratios, stage_{stage_idx}')
        plt.savefig(file_path)
        plt.show()
    '''
    '''
    for stage_idx in range(10):
        print(stage_idx)
        indices = np.linspace(start=0, stop=99, num=100, dtype=int)
        backbone_features, iid_scores, iid_predictions, iid_uncertainties, iid_features, mpn_features, mpn_scores, \
        mpn_predictions, mpn_uncertainties, targets = \
            active_visualizer.evaluate_model_on_custom_batch(stage=stage_idx, indices=indices, use_test_set=True)

        graph_builder = DenseGraphBuilder(GraphBuilderHyperParameterSet(NoEdgeAttributeDefinitionSet()))
        _, edge_index = graph_builder.get_graph(iid_features)
        after_mpn_features = active_visualizer._get_model(
            stage=stage_idx).eval().model.mpn_net.model.mpn_layers.layer_0.trans.forward(iid_features, edge_index)

        skip_vects, _ = active_visualizer.inspect_model_on_custom_batch(stage=stage_idx, indices=indices,
                                                                        layers_to_inspect=[
                                                                            "model.mpn_net.model.mpn_layers.layer_0.trans.lin_skip"],
                                                                        use_test_dataset=True)
        skip_vects = skip_vects["model.mpn_net.model.mpn_layers.layer_0.trans.lin_skip"].squeeze()

        final_value_vects = after_mpn_features - skip_vects

        the_model = active_visualizer._get_model(stage=stage_idx)

        for sample_idx in range(100):
            current_skip_vect = skip_vects[sample_idx, :].unsqueeze(dim=1)
            current_final_value_vect = final_value_vects[sample_idx, :].unsqueeze(dim=1)
            ##### Special:
            current_norm_ratio = torch.linalg.norm(current_skip_vect[:, 0], ord=2, dim=0)/torch.linalg.norm(current_final_value_vect[:, 0], ord=2, dim=0)
            current_final_value_vect *= current_norm_ratio
            #####
            predicted_probabilities = torch.zeros((200, 10))
            weights = torch.linspace(start=0.01, end=2, steps=200)
            for i, weight in enumerate(weights):
                current_input = current_skip_vect + weight * current_final_value_vect
                current_input = current_input.T
                x2 = current_input
                repeated_iid_features = iid_features[sample_idx].repeat(1, 1)
                x = repeated_iid_features + x2
                x = the_model.model.mpn_net.model.mpn_layers.layer_0.norm1(x)
                x = the_model.model.mpn_net.model.mpn_layers.layer_0.linear2(the_model.model.mpn_net.model.mpn_layers.layer_0.dropout(the_model.model.mpn_net.model.mpn_layers.layer_0.activation(the_model.model.mpn_net.model.mpn_layers.layer_0.linear1(x))))
                x = x + the_model.model.mpn_net.model.mpn_layers.layer_0.dropout2(x)
                x = the_model.model.mpn_net.model.mpn_layers.layer_0.norm2(x)
                scores = the_model.model.mpn_net.head(x).squeeze()
                probs = torch.nn.functional.softmax(scores, dim=0)
                predicted_probabilities[i, :] = probs

            predicted_probabilities = predicted_probabilities.cpu().detach().numpy()
            for class_idx in range(10):
                plt.plot(weights, predicted_probabilities[:, class_idx], label=f"class_{class_idx}")
                plt.text(x=1.6, y=0.1, s=f"true_class={targets[sample_idx]}")

            plt.legend()
            plt.xlabel("final value vect weight")
            plt.ylabel("probability")
            stage_folder_path = active_visualizer._get_logger_folder_path(stage=stage_idx)
            if not os.path.exists(os.path.join(stage_folder_path, 'deep_analysis')):
                os.mkdir(os.path.join(stage_folder_path, 'deep_analysis'))
            plt.title(f"EQUAL SIZE predicted class probabilities \n as a function of final value vect weight, sample_{sample_idx}")
            file_path = os.path.join(stage_folder_path, 'deep_analysis/'
                                     f'EQUAL_SIZE_prediction_change_due_to_MPN_stage_{stage_idx}_sample_{sample_idx}')
            plt.savefig(file_path)
            plt.close()
    '''
    '''
    results = []
    indices = np.linspace(start=0, stop=99, num=100, dtype=int)
    backbone_features, iid_scores, iid_predictions, iid_uncertainties, iid_features, mpn_features, mpn_scores, \
    mpn_predictions, mpn_uncertainties, targets = \
        active_visualizer.evaluate_model_on_custom_batch(stage=0, indices=indices, use_test_set=True)

    mpn_predictions = mpn_predictions.cpu().detach().numpy().squeeze()
    mpn_predictions = np.argmax(mpn_predictions, axis=1).astype(dtype=int)

    most_influential_samples = np.zeros((100, 4), dtype=int)
    predictions_on_most_influential_samples = np.zeros((100, 4), dtype=int)
    for i in range(4):
        most_influential_samples[:, i] = active_visualizer.visualize_attention_weights_of_a_batch(stage=0, indices=indices, head_idx=i, use_test_set=True).astype(dtype=int)
        predictions_on_most_influential_samples[:, i] = mpn_predictions[most_influential_samples[:, i]]

    for sample_idx in range(100):
        sample_results = {}
        for head_idx in range(4):
            sample_results[head_idx] = [most_influential_samples[sample_idx, head_idx], predictions_on_most_influential_samples[sample_idx, head_idx]]
        results.append(sample_results)

    print(results)
     '''
    #active_visualizer.visualize_attention_weights_of_a_batch(stage=0, indices=np.linspace(0, 99, 100, dtype=int), head_idx=3, use_test_set=True)




