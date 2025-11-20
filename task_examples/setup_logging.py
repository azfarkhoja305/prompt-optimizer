### This is AI generated code

import logging
from typing import Iterable, Tuple


class PackagePrefixFilter(logging.Filter):
    """Allow records whose logger name is under ANY of the given package prefixes."""

    def __init__(self, prefixes: Iterable[str]):
        super().__init__()
        self.prefixes: Tuple[str, ...] = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        n = record.name
        # allow exact match or dotted descendants
        return any(n == p or n.startswith(p + ".") for p in self.prefixes)


def setup_logging(level=logging.INFO):
    # Attach a single filtered handler to the ROOT so it catches all records,
    # but passes through only the prefixes we care about.
    root = logging.getLogger()
    root.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    # Only keep logs from our packages
    ch.addFilter(PackagePrefixFilter(("prompt_optimizer", "task_examples", "__main__")))

    # Avoid duplicate handlers if setup is called twice
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(ch)
