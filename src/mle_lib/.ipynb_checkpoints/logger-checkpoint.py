import logging
import logging.config
import os

from .config import log_path

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}


def configure_logger(
    logger=None,
    cfg=LOGGING_DEFAULT_CONFIG,
    log_file=None,
    console=True,
    log_level="DEBUG",
):
    """
    Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
    logger:
            Predefined logger object if present. If None a ew logger object will be created from root.
    cfg: dict()
            Configuration of the logging to be implemented by default
    log_file: str
            Path to the log file for logs to be stored
    console: bool
            To include a console handler(logs printing in console)
    log_level: str
            One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
            default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """

    logging.config.dictConfig(cfg)

    if not os.path.exists(log_file):
        os.makedirs(os.path.split(log_file)[0], exist_ok=True)
        with open(log_file, "w") as fp:
            pass

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


logger = configure_logger(log_file=log_path)
logger.info(
    "-------------------------------------------------------------------------------------------------------"
)

