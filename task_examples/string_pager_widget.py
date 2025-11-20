# ruff: noqa
# mypy: ignore-errors

### This is AI generated code

import ipywidgets as widgets
from IPython.display import display
import html


class StringPager:
    def __init__(self, items, *, wrap=False, layout=None):
        self.items = list(items)
        self.wrap = bool(wrap)
        self.index = 0
        if layout is None:
            layout = widgets.Layout(width="60%")

        # UI
        self.prev_btn = widgets.Button(description="⟨ Prev")
        self.next_btn = widgets.Button(description="Next ⟩")
        self.pos_lbl = widgets.Label()
        self.body = widgets.HTML()

        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)

        controls = widgets.HBox([self.prev_btn, self.pos_lbl, self.next_btn])

        if isinstance(layout, widgets.Layout):
            self._container = widgets.VBox([controls, self.body], layout=layout)
        else:
            self._container = widgets.VBox([controls, self.body])

        self._update()

    def _update(self):
        n = len(self.items)
        if n == 0:
            self.pos_lbl.value = "0 / 0"
            self.body.value = "<em>No items</em>"
            self.prev_btn.disabled = True
            self.next_btn.disabled = True
            return

        self.index = max(0, min(self.index, n - 1))
        self.pos_lbl.value = f"{self.index + 1} / {n}"
        text = html.escape(str(self.items[self.index]))
        self.body.value = f"<pre style='margin:0; white-space:pre-wrap'>{text}</pre>"

        self.prev_btn.disabled = (not self.wrap) and (self.index <= 0)
        self.next_btn.disabled = (not self.wrap) and (self.index >= n - 1)

    def _on_prev(self, _):
        if not self.items:
            return
        if self.index > 0:
            self.index -= 1
        elif self.wrap:
            self.index = len(self.items) - 1
        self._update()

    def _on_next(self, _):
        if not self.items:
            return
        if self.index < len(self.items) - 1:
            self.index += 1
        elif self.wrap:
            self.index = 0
        self._update()

    def display(self):
        display(self._container)

    def set_items(self, items, start=0):
        self.items = list(items)
        self.index = int(start) if self.items else 0
        self._update()

    @property
    def value(self):
        return None if not self.items else self.items[self.index]

    @property
    def widget(self):
        return self._container
