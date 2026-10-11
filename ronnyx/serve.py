import logging

import uvicorn
from uvicorn.logging import DefaultFormatter

LOGGER_NAME = "ronnyx"


def configure_logging() -> None:
    """Configure Ronnyx startup logs alongside Uvicorn's default output."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(DefaultFormatter("%(levelprefix)s %(message)s"))
    logger.addHandler(handler)


def main():
    configure_logging()
    uvicorn.run("ronnyx.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
