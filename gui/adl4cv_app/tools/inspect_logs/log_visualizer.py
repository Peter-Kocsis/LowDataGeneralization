import logging
import os
from abc import abstractmethod
from collections import defaultdict
from typing import List, Optional, Type, Union

import numpy as np
import plotly
from h2o_wave import ui

from gui.adl4cv_app.tools.common.tensorboard_view import TensorBoardView
from lowdataregime.parameters.log_loader import ActiveLearningLogLoader, ActiveLearningBenchmarkLog, \
    ActiveLearningTrainingLog, ActiveLearningRunLog, ActiveLearningScopeLog, ActiveLearningScope, ActiveLearningStageLog
from gui.adl4cv_app.tools.tensorboard.tensorboard import TensorBoardProcess
from gui.adl4cv_app.utils import View

import plotly.io as pio
import pandas as pd

import plotly.graph_objects as go


class LogScopeView(View):

    def __init__(self, parent: Optional[Type[View]], view_id: str, log_loader: ActiveLearningLogLoader,
                 view_attribs: List[str] = None):
        super().__init__(parent, view_id, view_attribs)
        self.log_loader = log_loader
        self.scope_logs: Optional[ActiveLearningScopeLog] = self._get_scope_logs()

    @property
    @abstractmethod
    def scope(self) -> ActiveLearningScope:
        raise NotImplementedError()

    def _get_scope_logs(self):
        return self.log_loader.get_scope_logs(self.scope)

    def _get_breadcrumb_items(self):
        breadcrumb_items = []
        if self.parent is not None:
            breadcrumb_items.extend(self.parent._get_breadcrumb_items())

        breadcrumb_items.append(
            ui.breadcrumb(
                name=f"#{self.route}",
                label=self.name)
        )
        return breadcrumb_items

    def _get_breadcrumbs_card(self):
        return ui.breadcrumbs_card(
            box='3 1 -1 1',
            items=self._get_breadcrumb_items(),
            commands=[
                ui.command(name='download', label='Download logs', icon='Download'),
            ]
        )


class RootScopeView(LogScopeView):
    @property
    def name(self):
        return 'Active Learning'

    @property
    def scope(self):
        return ActiveLearningScope()

    # noinspection PyMethodOverriding
    def from_attributes(self, log_loader: ActiveLearningLogLoader, tool_attribs: Optional[List[str]] = None,
                        view_attribs: Optional[List[str]] = None, *args, **kwargs):
        tool_attribs = tool_attribs or []
        if len(tool_attribs) == 0:
            return self
        else:
            view_id, tool_attribs = tool_attribs[0], tool_attribs[1:]
            child_view = DataSetScopeView(self, view_id, log_loader, view_attribs)
            return child_view.from_attributes(log_loader, tool_attribs, view_attribs, *args, **kwargs)

    async def show(self, q):
        gui_elements = {
            "breadcrumbs_card": self._get_breadcrumbs_card(),
            "context_buttons": ui.form_card(box='3 2 -1 -1', items=self.__get_context_buttons())
        }
        self.add_gui_elements(q, gui_elements)

    def __get_context_buttons(self):
        buttons = [ui.button(name=f"#{self.route}/{data_id}", label=data_id) for data_id in self.scope_logs.data_ids]
        return [ui.buttons(buttons[i:i + 4]) for i in range(0, len(buttons), 4)]


class DataSetScopeView(LogScopeView):
    @property
    def name(self):
        return self.view_id

    @property
    def scope(self) -> ActiveLearningScope:
        return ActiveLearningScope(data_id=self.view_id)

    # noinspection PyMethodOverriding
    def from_attributes(self, log_loader: ActiveLearningLogLoader, tool_attribs: Optional[List[str]] = None,
                        view_attribs: Optional[List[str]] = None, *args, **kwargs):
        tool_attribs = tool_attribs or []
        if len(tool_attribs) == 0:
            return self
        else:
            view_id, tool_attribs = tool_attribs[0], tool_attribs[1:]
            child_view = ModelScopeView(self, view_id, log_loader, view_attribs)
            return child_view.from_attributes(log_loader, tool_attribs, view_attribs, *args, **kwargs)

    def squeeze_into_pandas(self, panda_dict, column):
        layer_wise_data = {id: df.T[column] for id, df in panda_dict.items()}
        # TODO: Implement it generally
        layer_wise_data.update({'size_of_labeled_pool': list(range(11))})
        result = pd.DataFrame.from_dict(layer_wise_data)
        self.logger.debug(f"Squeezed panda: {result}")
        return result

    async def show(self, q):
        self.logger.debug(f"Creating plot from results: {self.scope_logs.results.columns}")

        gui_elements = {
            "breadcrumbs_card": self._get_breadcrumbs_card(),
            "context_buttons": ui.form_card(box='3 8 -1 -1', items=self.__get_context_buttons()),
            "result_visualization":
                await ResultView.get_view(q, self.view_attribs, self.route, self.scope_logs.results,
                                          self.scope_logs.get_columns_by_segment(self.scope_logs.results),
                                          '3 2 -1 6', legend_on_the_left=True),
            "max_attention_visualization":
                await ResultView.get_view(q, self.view_attribs, self.route, self.scope_logs.results,
                                          self.scope_logs.get_columns_by_segment(segment="attention_weights_max"),
                                          '13 2 5 8'),
            "std_attention_visualization":
                await ResultView.get_view(q, self.view_attribs, self.route, self.scope_logs.results,
                                          self.scope_logs.get_columns_by_segment(segment="attention_weights_std"),
                                          '18 2 5 8')
        }

        try:
            sample_occurences = self.scope_logs.sample_occurences
            gui_elements.update(
                {f"sample_occurences_{occurence + 1}":
                     await ResultView.get_view(q, self.view_attribs, self.route,
                                               self.squeeze_into_pandas(sample_occurences, occurence),
                                               sample_occurences.keys(),
                                               f'{4 * occurence + 1} 12 4 10', legend_on_the_left=False,
                                               title=f"Num of samples with occurence {occurence + 1}")
                 for occurence in range(10)})
        except (KeyError, ValueError):
            pass

        self.add_gui_elements(q, gui_elements)

    def __get_context_buttons(self):
        buttons = [ui.button(name=f"#{self.route}/{model_id}", label=model_id) for model_id in
                   self.scope_logs.model_ids]
        return [ui.buttons(buttons[i:i + 4]) for i in range(0, len(buttons), 4)]


class ModelScopeView(LogScopeView):
    @property
    def name(self):
        return self.view_id

    @property
    def scope(self) -> ActiveLearningScope:
        return ActiveLearningScope(data_id=self.parent.view_id, model_id=self.view_id)

    # noinspection PyMethodOverriding
    def from_attributes(self, log_loader: ActiveLearningLogLoader, tool_attribs: Optional[List[str]] = None,
                        view_attribs: Optional[List[str]] = None, *args, **kwargs):
        tool_attribs = tool_attribs or []
        if len(tool_attribs) == 0:
            return self
        else:
            view_id, tool_attribs = tool_attribs[0], tool_attribs[1:]
            child_view = self.get_child_view(view_id, view_attribs)
            return child_view.from_attributes(log_loader, tool_attribs, view_attribs, *args, **kwargs)

    def get_child_view(self, view_id, view_attribs):
        active_learning_log = self.log_loader.active_learning_logs[view_id]
        self.logger.debug(f"Active learning log {active_learning_log} ({type(active_learning_log)}) selected!")
        if isinstance(active_learning_log, ActiveLearningBenchmarkLog):
            return BenchmarkView(self, view_id, active_learning_log, view_attribs)
        else:
            return TrainingView(self, view_id, active_learning_log, view_attribs)

    async def show(self, q):
        gui_elements = {
            "breadcrumbs_card": self._get_breadcrumbs_card(),
            "context_buttons": ui.form_card(box='3 8 -1 -1', items=self.__get_context_buttons()),
            "result_visualization":
                await ResultView.get_view(q, self.view_attribs, self.route, self.scope_logs.results,
                                          self.scope_logs.get_columns_by_segment(self.scope_logs.results),
                                          '3 2 -1 6', legend_on_the_left=True)
        }
        self.add_gui_elements(q, gui_elements)

    def __get_context_buttons(self):
        buttons = [ui.button(name=f"#{self.route}/{log_id}", label=log_id) for log_id in self.scope_logs.log_ids]
        return [ui.buttons(buttons[i:i + 4]) for i in range(0, len(buttons), 4)]


class BenchmarkView(View):
    def __init__(self, parent: Optional[Type[View]], view_id: str, benchmark_log: ActiveLearningBenchmarkLog,
                 view_attribs: List[str] = None):
        super().__init__(parent, view_id, view_attribs)
        self.benchmark_log = benchmark_log
        self.tensorboard_process: Optional[TensorBoardProcess] = None

    @property
    def name(self):
        return self.view_id

    # noinspection PyMethodOverriding
    def from_attributes(self, log_loader: ActiveLearningLogLoader, tool_attribs: Optional[List[str]] = None,
                        view_attribs: Optional[List[str]] = None, *args, **kwargs):
        tool_attribs = tool_attribs or []
        if len(tool_attribs) == 0:
            return self
        else:
            view_id, tool_attribs = tool_attribs[0], tool_attribs[1:]
            active_learning_log = self.benchmark_log.run_logs[int(view_id)]
            child_view = RunView(self, view_id, active_learning_log, view_attribs)
            return child_view.from_attributes(log_loader, tool_attribs, view_attribs, *args, **kwargs)

    async def show(self, q):
        gui_elements = {
            "breadcrumbs_card": self._get_breadcrumbs_card(),
            "run_tabs": self._get_run_tabs_card(self.route),
            "params_markdown": self._get_params_markdown_ui('3 2 5 4'),
            "tensorboard_frame": self._get_tensorboard_frame('1 16 -1 11'),
            "data_distribution": await ClassDistributionView.get_view(q,
                                                                      self.route,
                                                                      self.benchmark_log.data_distribution,
                                                                      '3 6 10 5'),
            "total_data_distribution": await ClassDistributionView.get_view(q,
                                                                            self.route,
                                                                            self.benchmark_log.total_data_distribution,
                                                                            '3 11 10 5'),
            "result_visualization":
                await ResultView.get_view(q, self.view_attribs, self.route, self.benchmark_log.benchmark_results,
                                          self.benchmark_log.get_accuracy_columns(self.benchmark_log.benchmark_results),
                                          '8 2 5 4'),
            "max_attention_visualization":
                await ResultView.get_view(q, self.view_attribs, self.route, self.benchmark_log.benchmark_results,
                                          ["attention_weights_max"],
                                          '13 2 5 4'),
            "std_attention_visualization":
                await ResultView.get_view(q, self.view_attribs, self.route, self.benchmark_log.benchmark_results,
                                          ["attention_weights_std"],
                                          '18 2 5 4')
        }
        self.add_gui_elements(q, gui_elements)

    def drop(self, q):
        if self.tensorboard_process is not None:
            self.tensorboard_process.stop()
            self.tensorboard_process = None
        super().drop(q)

    def _get_tensorboard_frame(self, box):
        self.tensorboard_process = TensorBoardProcess(self.benchmark_log.log_folder)
        self.tensorboard_process.start()

        tensorboard_view = TensorBoardView(self, "tensorboard")
        return tensorboard_view.get_tensorboard_frame(box)

    def _get_breadcrumb_items(self):
        breadcrumb_items = self.parent._get_breadcrumb_items()

        breadcrumb_items.append(
            ui.breadcrumb(
                name=f"#{self.route}/{self.view_id}",
                label=self.view_id)
        )
        return breadcrumb_items

    def _get_breadcrumbs_card(self):
        return ui.breadcrumbs_card(
            box='3 1 5 1',
            items=self._get_breadcrumb_items(),
            commands=[
                ui.command(name='download', label='Download logs', icon='Download'),
            ]
        )

    def _get_run_tabs_card(self, value: str, box: str = '8 1 -1 1'):
        items = [ui.tab(name=f'#{self.route}', label='Overview')] + \
                [ui.tab(name=f'#{self.route}/{run_id}', label=str(run_id))
                 for run_id, run_log in self.benchmark_log.run_logs.items()]
        return ui.tab_card(
            box=box,
            items=items,
            value=f'#{value}',
            link=True,
        )

    def _get_params_markdown_ui(self, box):
        return ui.markdown_card(
            box=box,
            title='Parameters',
            content=self.benchmark_log.params_markdown,
        )


class TrainingView(View):
    def __init__(self, parent: Optional[Type[View]], view_id: str, training_log: ActiveLearningTrainingLog,
                 view_attribs: List[str] = None):
        super().__init__(parent, view_id, view_attribs)
        self.training_log = training_log

    @property
    def name(self):
        return self.view_id

    # noinspection PyMethodOverriding
    def from_attributes(self, log_loader: ActiveLearningLogLoader, tool_attribs: Optional[List[str]] = None,
                        view_attribs: Optional[List[str]] = None, *args, **kwargs):
        tool_attribs = tool_attribs or []
        if len(tool_attribs) == 0:
            return self
        else:
            view_id, tool_attribs = tool_attribs[0], tool_attribs[1:]
            active_learning_log = self.training_log.stage_logs[int(view_id)]
            child_view = StageView(self, view_id, active_learning_log, view_attribs)
            return child_view.from_attributes(log_loader, tool_attribs, view_attribs, *args, **kwargs)

    async def show(self, q):
        gui_elements = {
            **self.get_navigation_header(self.route),
            "params_markdown": self._get_params_markdown_ui('3 2 5 4'),
            "tensorboard_frame": self._get_tensorboard_frame('1 16 -1 11'),
            "data_distribution": await ClassDistributionView.get_view(q,
                                                                      self.route,
                                                                      self.training_log.data_distribution,
                                                                      '3 6 10 3'),
            "total_data_distribution": await ClassDistributionView.get_view(q,
                                                                            self.route,
                                                                            self.training_log.total_data_distribution,
                                                                            '3 9 10 2'),
            "result_visualization": await ResultView.get_view(q, self.view_attribs, self.route,
                                                              self.training_log.results,
                                                              self.training_log.get_accuracy_columns(),
                                                              '8 2 5 4'),
            "attention_histogram": await AttentionHistogramView.get_view(q,
                                                                         self.route,
                                                                         self.training_log.attention_histogram,
                                                                         '1 11 12 5'),
        }
        self.add_gui_elements(q, gui_elements)

    def _get_tensorboard_frame(self, box):
        self.tensorboard_process = TensorBoardProcess(self.training_log.log_folder)
        self.tensorboard_process.start()

        tensorboard_view = TensorBoardView(self, "tensorboard")
        return tensorboard_view.get_tensorboard_frame(box)

    def get_navigation_header(self, value):
        return {
            "breadcrumbs_card": self._get_breadcrumbs_card(),
            "stage_tabs": self._get_stage_tabs_card(value)}

    def _get_breadcrumb_items(self):
        breadcrumb_items = self.parent._get_breadcrumb_items()

        breadcrumb_items.append(
            ui.breadcrumb(
                name=f"#{self.route}/{self.view_id}",
                label=self.view_id)
        )
        return breadcrumb_items

    def _get_breadcrumbs_card(self):
        return ui.breadcrumbs_card(
            box='3 1 5 1',
            items=self._get_breadcrumb_items(),
            commands=[
                ui.command(name='download', label='Download logs', icon='Download'),
            ]
        )

    def _get_stage_tabs_card(self, value):
        items = [ui.tab(name=f'#{self.route}', label='Overview')] + \
                [ui.tab(name=f'#{self.route}/{stage_id}', label=str(stage_id))
                 for stage_id, stage_log in self.training_log.stage_logs.items()]
        return ui.tab_card(
            box='8 1 -1 1',
            items=items,
            value=f'#{value}',
            link=True,
        )

    def _get_params_markdown_ui(self, box):
        return ui.markdown_card(
            box=box,
            title='Parameters',
            content=self.training_log.params_markdown,
        )


class RunView(View):
    def __init__(self, parent: BenchmarkView, view_id: str, run_log: ActiveLearningRunLog,
                 view_attribs: List[str] = None):
        super().__init__(parent, view_id, view_attribs)
        self.run_log = run_log

    @property
    def name(self):
        return self.view_id

    # noinspection PyMethodOverriding
    def from_attributes(self, log_loader: ActiveLearningLogLoader, tool_attribs: Optional[List[str]] = None,
                        view_attribs: Optional[List[str]] = None, *args, **kwargs):
        tool_attribs = tool_attribs or []
        if len(tool_attribs) == 0:
            return self
        else:
            view_id, tool_attribs = tool_attribs[0], tool_attribs[1:]
            active_learning_log = self.run_log.stage_logs[int(view_id)]
            child_view = StageView(self, view_id, active_learning_log, view_attribs)
            return child_view.from_attributes(log_loader, tool_attribs, view_attribs, *args, **kwargs)

    async def show(self, q):
        gui_elements = {
            **self.get_navigation_header(self.route),
            "params_markdown": self.parent._get_params_markdown_ui('3 2 5 4'),
            "data_distribution": await ClassDistributionView.get_view(q, self.route, self.run_log.data_distribution,
                                                                      '3 6 10 3'),
            "total_data_distribution": await ClassDistributionView.get_view(q,
                                                                            self.route,
                                                                            self.run_log.total_data_distribution,
                                                                            '3 9 10 2'),
            "result_visualization": await ResultView.get_view(q, self.view_attribs, self.route,
                                                              self.run_log.results, self.run_log.get_accuracy_columns(),
                                                              '8 2 5 4'),
            "attention_histogram": await AttentionHistogramView.get_view(q,
                                                                         self.route,
                                                                         self.run_log.attention_histogram,
                                                                         '1 11 12 5'),
        }
        self.add_gui_elements(q, gui_elements)

    def get_navigation_header(self, value):
        return {
            "breadcrumbs_card": self.parent._get_breadcrumbs_card(),
            "run_tabs": self.parent._get_run_tabs_card(value, box='8 1 2 1'),
            "stage_tabs": self._get_stage_tabs_card(value, box='10 1 -1 1')}

    def _get_stage_tabs_card(self, value: str, box: str = '10 1 -1 1'):
        items = [ui.tab(name=f'#{self.route}', label='Overview')] + \
                [ui.tab(name=f'#{self.route}/{stage_id}', label=str(stage_id))
                 for stage_id, stage_log in self.run_log.stage_logs.items()]
        return ui.tab_card(
            box=box,
            items=items,
            value=f'#{value}',
            link=True,
        )


class StageView(View):
    def __init__(self, parent: Union[RunView, TrainingView], view_id: str, stage_log: ActiveLearningStageLog,
                 view_attribs: List[str] = None):
        super().__init__(parent, view_id, view_attribs)
        self.stage_log = stage_log

    @property
    def name(self):
        return self.view_id

    # noinspection PyMethodOverriding
    def from_attributes(self, log_loader: ActiveLearningLogLoader, tool_attribs: Optional[List[str]] = None,
                        view_attribs: Optional[List[str]] = None, *args, **kwargs):
        tool_attribs = tool_attribs or []
        if len(tool_attribs) == 0:
            return self
        else:
            raise NotImplementedError()

    async def show(self, q):
        gui_elements = {
            **self.parent.get_navigation_header(self.route),
            "data_distribution": await ClassDistributionView.get_view(q,
                                                                      self.route,
                                                                      self.stage_log.data_distribution,
                                                                      '3 2 10 4'),
            "attention_histogram": await AttentionHistogramView.get_view(q,
                                                                         self.route,
                                                                         self.stage_log.attention_histogram,
                                                                         '3 6 10 5'),
        }
        self.add_gui_elements(q, gui_elements)


class AttentionHistogramView:
    PLOTLY_CONFIG = {'displaylogo': False}
    PLOTLY_LAYOUT = {'margin': dict(l=10, r=10, t=10, b=10),
                     'paper_bgcolor': 'rgb(255, 255, 255)',
                     'plot_bgcolor': 'rgb(255, 255, 255)',
                     'hovermode': 'closest',
                     'xaxis': {'showspikes': True, 'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgb(225, 225, 225)'},
                     'yaxis': {'showspikes': True, 'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgb(225, 225, 225)'},
                     'hoverlabel': {'namelength': -1},
                     'barmode': "group",
                     # 'bargroupgap': 0.8  # gap between bars of the same location coordinate.
                     }
    logger = logging.getLogger('ClassDistributionView')

    @staticmethod
    async def get_view(
            q,
            route: str,
            attention_histogram: dict,
            box: str,
            legend_on_the_left: bool = True,
            title: str = "Attention histogram"):
        AttentionHistogramView.logger.debug(f"Creating attention histogram view from: {attention_histogram}")

        distribution_plot, distribution_file_path = await AttentionHistogramView.__get_histogram_plot(q,
                                                                                                      attention_histogram,
                                                                                                      legend_on_the_left)

        # WORKAROUND: Height setting is not working in H2OWave
        y_height = int(box.split(' ')[-1])
        y_px = y_height * 80
        return ui.form_card(title=title,
                            box=box,
                            items=[ui.frame(width='100%',
                                            height=f'{y_px}px',
                                            name='data distribution_plot',
                                            content=distribution_plot),
                                   ui.text(f'[Download plot]({distribution_file_path})')],
                            commands=[ui.command(name=f'#{route}#result_table_view',
                                                 label='Table view',
                                                 caption='Table view',
                                                 icon='Tiles')])

    @staticmethod
    async def __get_histogram_plot(q, histogram: dict, legend_on_the_left: bool = False):
        if legend_on_the_left:
            legend_layout = {
                'legend': {'orientation': "v", 'yanchor': "bottom", 'y': 0, 'xanchor': "right", 'x': -0.05}}
        else:
            legend_layout = {'legend': {'orientation': "h", 'yanchor': "top", 'y': -0.05, 'xanchor': "left", 'x': 0}}

        fig = AttentionHistogramView._get_plot(histogram)

        fig.update_layout(AttentionHistogramView.PLOTLY_LAYOUT)
        fig.update_layout(legend_layout)

        html_content = pio.to_html(fig, validate=False, config=ResultView.PLOTLY_CONFIG, include_plotlyjs='cdn')

        plot_file_path = await AttentionHistogramView._get_download_path(q, html_content)

        return html_content, plot_file_path

    @staticmethod
    async def _get_download_path(q, html_content):
        tmp_path = "attention_histogram.html"
        with open(tmp_path, "w") as tmp_file:
            tmp_file.write(html_content)
        plot_file_path, = await q.site.upload(["attention_histogram.html"])
        os.remove(tmp_path)
        return plot_file_path

    @staticmethod
    def _get_plot(attention_histogram):
        depth = AttentionHistogramView._get_depth(attention_histogram)
        AttentionHistogramView.logger.debug(f"Distribution depth: {depth}")
        if depth == -1:
            return AttentionHistogramView._get_stage_attention_histogram_plot(attention_histogram)
        if depth == 0:
            return AttentionHistogramView.__get_run_distribution_plot(attention_histogram)
        if depth == 1:
            raise NotImplementedError()
            # return AttentionHistogramView.__get_benchmark_distribution_plot(distribution)
        raise NotImplementedError()

    @staticmethod
    def _get_depth(attention_histogram):
        depth = -1
        while isinstance(attention_histogram, dict):
            depth += 1
            if len(attention_histogram) > 0:
                attention_histogram = next(iter(attention_histogram.values()))
            else:
                break
        return depth

    @staticmethod
    def _get_stage_attention_histogram_plot(attention_histogram: pd.DataFrame):
        fig = go.Figure()

        x = attention_histogram.index / len(attention_histogram.index)
        for column in attention_histogram.columns:
            fig.add_trace(go.Bar(x=x,
                                 y=attention_histogram[column],
                                 name=column))
        return fig

    @staticmethod
    def __get_run_distribution_plot(attention_histogram):
        fig = go.Figure()

        for stage_id, stage_attention_histogram in attention_histogram.items():
            x = stage_attention_histogram.index / len(stage_attention_histogram.index)
            for column in stage_attention_histogram.columns:
                fig.add_trace(go.Bar(x=x,
                                     y=stage_attention_histogram[column],
                                     name=f"{stage_id}_{column}",
                                     legendgroup=stage_id))
        return fig

    @staticmethod
    def __get_benchmark_distribution_plot(distribution):
        fig = go.Figure()

        for run_id, run_dist in distribution.items():
            previous_stage_dist = defaultdict(int)
            for (stage_id, stage_distribution), color in zip(sorted(run_dist.items(), key=lambda x: x[0]),
                                                             plotly.colors.qualitative.Dark24 * (
                                                                     len(run_dist) // 24 + 1)):
                stage_dist_change = {class_name: occurences - previous_stage_dist[class_name] for class_name, occurences
                                     in stage_distribution.items()}
                previous_stage_dist = stage_distribution
                fig.add_trace(
                    go.Bar(name=stage_id,
                           legendgroup=stage_id,
                           marker_color=color,
                           width=0.8,
                           offset=-(run_id - 1) * 0.2 - 0.4,
                           x=[list(stage_dist_change.keys()), [run_id] * len(stage_dist_change)],
                           y=list(stage_dist_change.values())))

        # for run_id, run_dist in distribution.items():
        #     previous_stage_dist = defaultdict(int)
        #     for (stage_id, stage_distribution), color in zip(sorted(run_dist.items(), key=lambda x: x[0]), plotly.colors.qualitative.Dark24 * (len(run_dist) // 24 + 1)):
        #         stage_dist_change = {class_name: occurences - previous_stage_dist[class_name] for class_name, occurences
        #                              in stage_distribution.items()}
        #         fig.add_trace(
        #             go.Bar(name=stage_id,
        #                    legendgroup=stage_id,
        #                    marker_color=color,
        #                    offsetgroup=run_id,
        #                    base=list(previous_stage_dist.values()),
        #                    x=list(stage_dist_change.keys()),
        #                    y=list(stage_dist_change.values())))
        #         previous_stage_dist = stage_distribution

        return fig


class ClassDistributionView:
    PLOTLY_CONFIG = {'displaylogo': False}
    PLOTLY_LAYOUT = {'margin': dict(l=10, r=10, t=10, b=10),
                     'paper_bgcolor': 'rgb(255, 255, 255)',
                     'plot_bgcolor': 'rgb(255, 255, 255)',
                     'hovermode': 'closest',
                     'xaxis': {'showspikes': True, 'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgb(225, 225, 225)'},
                     'yaxis': {'showspikes': True, 'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgb(225, 225, 225)'},
                     'hoverlabel': {'namelength': -1},
                     'barmode': "stack",
                     # 'bargroupgap': 0.8  # gap between bars of the same location coordinate.
                     }
    logger = logging.getLogger('ClassDistributionView')

    @staticmethod
    async def get_view(
            q,
            route: str,
            distribution: dict,
            box: str,
            legend_on_the_left: bool = True,
            title: str = "Data distribution"):
        ClassDistributionView.logger.debug(f"Creating distribution view from: {distribution}")

        distribution_plot, distribution_file_path = await ClassDistributionView.get_distribution_plot(q, distribution,
                                                                                                      legend_on_the_left)

        # WORKAROUND: Height setting is not working in H2OWave
        y_height = int(box.split(' ')[-1])
        y_px = y_height * 80
        return ui.form_card(title=title,
                            box=box,
                            items=[ui.frame(width='100%',
                                            height=f'{y_px}px',
                                            name='data distribution_plot',
                                            content=distribution_plot),
                                   ui.text(f'[Download plot]({distribution_file_path})')],
                            commands=[ui.command(name=f'#{route}#result_table_view',
                                                 label='Table view',
                                                 caption='Table view',
                                                 icon='Tiles')])

    @staticmethod
    async def get_distribution_plot(q, distribution: dict, legend_on_the_left: bool = False):
        if legend_on_the_left:
            legend_layout = {
                'legend': {'orientation': "v", 'yanchor': "bottom", 'y': 0, 'xanchor': "right", 'x': -0.05}}
        else:
            legend_layout = {'legend': {'orientation': "h", 'yanchor': "top", 'y': -0.05, 'xanchor': "left", 'x': 0}}

        fig = ClassDistributionView.get_plot(distribution)

        fig.update_layout(ClassDistributionView.PLOTLY_LAYOUT)
        fig.update_layout(legend_layout)

        html_content = pio.to_html(fig, validate=False, config=ResultView.PLOTLY_CONFIG, include_plotlyjs='cdn')

        plot_file_path = await ClassDistributionView._get_download_path(q, html_content)

        return html_content, plot_file_path

    @staticmethod
    async def _get_download_path(q, html_content):
        tmp_path = "distribution.html"
        with open(tmp_path, "w") as tmp_file:
            tmp_file.write(html_content)
        plot_file_path, = await q.site.upload(["distribution.html"])
        os.remove(tmp_path)
        return plot_file_path

    @staticmethod
    def get_plot(distribution):
        depth = ClassDistributionView._get_depth(distribution)
        ClassDistributionView.logger.debug(f"Distribution depth: {depth}")
        if depth == 0:
            return ClassDistributionView._get_stage_distribution_plot(distribution)
        if depth == 1:
            return ClassDistributionView.__get_run_distribution_plot(distribution)
        if depth == 2:
            return ClassDistributionView.__get_benchmark_distribution_plot(distribution)
        raise NotImplementedError()

    @staticmethod
    def _get_depth(distribution):
        depth = -1
        while isinstance(distribution, dict):
            depth += 1
            distribution = next(iter(distribution.values()))
        return depth

    @staticmethod
    def _get_stage_distribution_plot(distribution):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(distribution.keys()),
            y=list(distribution.values())))
        return fig

    @staticmethod
    def __get_run_distribution_plot(distribution):
        fig = go.Figure()

        previous_stage_dist = defaultdict(int)
        for (stage_id, stage_distribution), color in zip(sorted(distribution.items(), key=lambda x: x[0]),
                                                         plotly.colors.qualitative.Dark24 * (
                                                                 len(distribution) // 24 + 1)):
            stage_dist_change = {class_name: occurences - previous_stage_dist[class_name] for class_name, occurences in
                                 stage_distribution.items()}
            previous_stage_dist = stage_distribution
            fig.add_trace(
                go.Bar(name=stage_id,
                       x=list(stage_dist_change.keys()),
                       y=list(stage_dist_change.values()),
                       marker_color=color))
        return fig

    @staticmethod
    def __get_benchmark_distribution_plot(distribution):
        fig = go.Figure()

        for run_id, run_dist in distribution.items():
            previous_stage_dist = defaultdict(int)
            for (stage_id, stage_distribution), color in zip(sorted(run_dist.items(), key=lambda x: x[0]),
                                                             plotly.colors.qualitative.Dark24 * (
                                                                     len(run_dist) // 24 + 1)):
                stage_dist_change = {class_name: occurences - previous_stage_dist[class_name] for class_name, occurences
                                     in stage_distribution.items()}
                previous_stage_dist = stage_distribution
                fig.add_trace(
                    go.Bar(name=stage_id,
                           legendgroup=stage_id,
                           marker_color=color,
                           width=0.8,
                           offset=-(run_id - 1) * 0.2 - 0.4,
                           x=[list(stage_dist_change.keys()), [run_id] * len(stage_dist_change)],
                           y=list(stage_dist_change.values())))

        # for run_id, run_dist in distribution.items():
        #     previous_stage_dist = defaultdict(int)
        #     for (stage_id, stage_distribution), color in zip(sorted(run_dist.items(), key=lambda x: x[0]), plotly.colors.qualitative.Dark24 * (len(run_dist) // 24 + 1)):
        #         stage_dist_change = {class_name: occurences - previous_stage_dist[class_name] for class_name, occurences
        #                              in stage_distribution.items()}
        #         fig.add_trace(
        #             go.Bar(name=stage_id,
        #                    legendgroup=stage_id,
        #                    marker_color=color,
        #                    offsetgroup=run_id,
        #                    base=list(previous_stage_dist.values()),
        #                    x=list(stage_dist_change.keys()),
        #                    y=list(stage_dist_change.values())))
        #         previous_stage_dist = stage_distribution

        return fig


class ResultView:
    PLOTLY_CONFIG = {'displaylogo': False}
    PLOTLY_LAYOUT = {'margin': dict(l=10, r=10, t=10, b=10),
                     'paper_bgcolor': 'rgb(255, 255, 255)',
                     'plot_bgcolor': 'rgb(255, 255, 255)',
                     'hovermode': 'closest',
                     'xaxis': {'showspikes': True, 'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgb(225, 225, 225)'},
                     'yaxis': {'showspikes': True, 'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgb(225, 225, 225)'},
                     'hoverlabel': {'namelength': -1}}

    @staticmethod
    async def get_view(
            q,
            view_attribs: List[str],
            route: str,
            results: pd.DataFrame,
            y: List[str],
            box: str,
            legend_on_the_left: bool = False,
            title: str = "Active Learning results"):

        try:
            if len(view_attribs) > 0:
                if view_attribs[0] == "result_table_view":
                    return ui.form_card(box=box, items=[ResultView.__get_results_table(results)],
                                        commands=[ui.command(name=f'#{route}',
                                                             label='Plot view',
                                                             caption='Plot view',
                                                             icon='Tiles')])
            else:
                result_plot, plot_file_path = await ResultView.get_results_plot(q, results, y, legend_on_the_left)

                # WORKAROUND: Height setting is not working in H2OWave
                y_height = int(box.split(' ')[-1])
                y_px = y_height * 80
                return ui.form_card(title=title,
                                    box=box, items=[ui.frame(width='100%', height=f'{y_px}px',
                                                             name='active_learning_plot',
                                                             content=result_plot),
                                                    ui.text(f'[Download plot]({plot_file_path})')],
                                    commands=[ui.command(name=f'#{route}#result_table_view',
                                                         label='Table view',
                                                         caption='Table view',
                                                         icon='Tiles')])
        except Exception:
            return ui.form_card(title=title,
                                box=box,
                                items=[ui.text('Not found!')])

    @staticmethod
    def __get_results_table(results: pd.DataFrame):
        return ui.table(name='results_table',
                        columns=[ui.table_column(name=column, label=column) for column in
                                 results.columns],
                        rows=[ui.table_row(name=str(idx), cells=[str(val) for val in row])
                              for idx, row in results.iterrows()],
                        downloadable=True)

    @staticmethod
    async def get_results_plot(q, results: pd.DataFrame, y: List[str], legend_on_the_left: bool = False):
        if legend_on_the_left:
            legend_layout = {
                'legend': {'orientation': "v", 'yanchor': "bottom", 'y': 0, 'xanchor': "right", 'x': -0.05}}
        else:
            legend_layout = {'legend': {'orientation': "h", 'yanchor': "top", 'y': -0.05, 'xanchor': "left", 'x': 0}}

        fig = go.Figure()
        for column, color in zip(y, plotly.colors.qualitative.Dark24 * (len(y) // 24 + 1)):
            if column.endswith("_std"):
                continue

            x = results["size_of_labeled_pool"]
            y = results[column]

            std_column = f"{column}_std"

            error_y = None
            if std_column in results:
                error_y = dict(
                    type='data',  # value of error bar given in data coordinates
                    array=results[std_column],
                    visible=True)

            fig.add_trace(go.Scatter(x=x,
                                     y=y,
                                     error_y=error_y,
                                     mode='lines+markers',
                                     name=column,
                                     line=dict(color=color)))

        fig.update_layout(ResultView.PLOTLY_LAYOUT)
        fig.update_layout(legend_layout)

        html_content = pio.to_html(fig, validate=False, config=ResultView.PLOTLY_CONFIG, include_plotlyjs='cdn')

        tmp_path = "plot.html"
        with open(tmp_path, "w") as tmp_file:
            tmp_file.write(html_content)
        plot_file_path, = await q.site.upload(["plot.html"])
        os.remove(tmp_path)

        return html_content, plot_file_path
