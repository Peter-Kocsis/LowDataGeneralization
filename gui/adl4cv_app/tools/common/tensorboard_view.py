from typing import Optional, Type, List
from h2o_wave import ui

from gui.adl4cv_app.utils import View


class TensorBoardView(View):
    def __init__(self, parent: Optional[Type[View]], view_id: str,
                 view_attribs: List[str] = None):
        super().__init__(parent, view_id, view_attribs)

    @property
    def name(self):
        return "TensorBoard"

    @classmethod
    def from_attributes(cls, tool_attribs: Optional[List[str]] = None, view_attribs: Optional[List[str]] = None, *args,
                        **kwargs) -> Type['View']:
        raise NotImplementedError()

    async def show(self, q):
        gui_elements = {
            "tensorboard_frame": self.get_tensorboard_frame('3 1 -1 10')
        }
        self.add_gui_elements(q, gui_elements)

    def get_tensorboard_frame(self, box):
        y_height = int(box.split(' ')[-1])
        y_px = y_height * 85
        return ui.form_card(
            box=box,
            items=[ui.frame(path='http://localhost:6006/', width='100%', height=f'{y_px}px')]
        )
