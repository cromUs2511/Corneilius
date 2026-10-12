"""Corneilius Qt entry point. The CustomTkinter UI remains available via jarvis.main."""

from __future__ import annotations

import logging

from jarvis.config import Settings
from jarvis.interface.qt_app import build_qt_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler()])
    app, window = build_qt_app(Settings())
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
