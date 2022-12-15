import logging
import threading
from abc import abstractmethod, ABC
from typing import List, Optional, Type

from h2o_wave import ui
from h2o_wave.core import AsyncPage, AsyncSite


# class StoppableThread(threading.Thread):
#     def __init__(self, sleep_time_sec, target, **kwargs):
#         super(StoppableThread, self).__init__(target=target, **kwargs)
#         self.setDaemon(True)
#         self.stop_event = threading.Event()
#         self.sleep_time = sleep_time_sec
#
#         if target is None:
#             raise Exception('No target function given')
#
#         self.target = target
#
#     def stop(self):
#         self.stop_event.set()
#
#     def stopped(self):
#         return self.stop_event.isSet()
#
#     def run(self):
#         while not self.stopped():
#             self.target()
#             self.stop_event.wait(self.sleep_time)


class Atomic(object):
    def __init__(self, value):
        self._lock = threading.Lock()
        self._value = value

    @property
    def value(self):
        with self._lock:
            ret_val =  self._value
        return ret_val

    @value.setter
    def value(self, value):
        with self._lock:
            self._value = value

    def run(self, target):
        with self._lock:
            target(self._value)

    async def async_run(self, async_target):
        with self._lock:
            await async_target(self._value)


class StoppableThread(threading.Thread):
    def __init__(self, target, **kwargs):
        super(StoppableThread, self).__init__(target=target, **kwargs)
        self.setDaemon(True)
        self.stop_event = threading.Event()

        if target is None:
            raise Exception('No async_target function given')

        self.target = target

    def stop(self):
        self.stop_event.set()

    def stopped(self):
        return self.stop_event.isSet()


class StoppablePollingThread(StoppableThread):
    def __init__(self, sleep_time_sec, target, **kwargs):
        super().__init__(target=target, **kwargs)
        self.sleep_time = sleep_time_sec

        if target is None:
            raise Exception('No async_target function given')

        self.target = target

    def run(self):
        while not self.stopped():
            self.target()
            self.stop_event.wait(self.sleep_time)


class AsyncUI(ABC, StoppableThread):
    def __init__(self, q, sleep_time_sec=5):
        StoppableThread.__init__(self, sleep_time_sec=sleep_time_sec, target=self.update)
        self.q = q
        self.sleep_time_sec = sleep_time_sec

        self.async_site = AsyncSite()

    @property
    @abstractmethod
    def async_page(self) -> AsyncPage:
        pass

    @property
    @abstractmethod
    def initial_async_view(self):
        pass

    @abstractmethod
    def update(self):
        pass

    async def serve(self):
        if not self.q.client.initialized:
            # Set up up the page at /stats
            self.async_page.drop()  # Clear any existing page
            self.async_page['example'] = self.initial_async_view
            await self.async_page.save()

            # Set up this app's UI
            self.q.page['form'] = ui.form_card(box='1 1 -1 -1', items=[
                ui.frame(path=self.async_page.url, height='800px'),
            ])
            await self.q.page.save()

            self.q.client.initialized = True

        await self.run()


class ADL4CVTool(ABC):
    ID = "undefined_tool_id"
    NAME = "undefined_tool_name"

    def __init__(self, q):
        self.q = q

        self._root_view: Optional[Type[View]] = None
        self.current_view: Optional[Type[View]] = None

    @property
    @abstractmethod
    def root_view(self):
        raise NotImplementedError()

    @abstractmethod
    def start(self, tool_attribs, view_attribs):
        pass

    def stop(self):
        self.current_view.drop(self.q.value)


class View(ABC):

    def __init__(self, parent: Optional[Type['View']], view_id: str, view_attribs: List[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.parent = parent
        self.view_id = view_id
        self.view_attribs = view_attribs or []
        self.gui_keys = set()

        self._route: Optional[str] = None

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @property
    def route(self) -> str:
        if self._route is None:
            if self.parent is None:
                self._route = self.view_id
            else:
                self._route = f"{self.parent.route}/{self.view_id}"
        return self._route

    @classmethod
    @abstractmethod
    def from_attributes(cls, tool_attribs: Optional[List[str]] = None, view_attribs: Optional[List[str]] = None, *args, **kwargs) -> Type['View']:
        raise NotImplementedError()

    @abstractmethod
    async def show(self, q):
        raise NotImplementedError()

    def drop(self, q):
        for gui_key in self.gui_keys:
            del q.page[gui_key]

    def _add_gui_element(self, q, name, gui_element):
        self.gui_keys.add(name)
        return q.page.add(key=name, card=gui_element)

    def add_gui_elements(self, q, gui_element_dict):
        gui_elements = dict()
        for name, gui_element in gui_element_dict.items():
            gui_elements[name] = self._add_gui_element(q, name, gui_element)
        return gui_elements


