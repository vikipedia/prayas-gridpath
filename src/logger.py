import logging
import logging.handlers
import os
import datetime


levels = {"INFO": logging.INFO,
          "WARN": logging.WARN,
          "DEBUG": logging.DEBUG,
          "ERROR": logging.ERROR}


def get_log_filepath(prefix):
    location = os.getcwd()
    format_ = "%Y-%m-%d-%H-%M-%S"
    timestamp = datetime.datetime.now().strftime(format_)
    return os.path.join(location, ".".join([prefix, timestamp, "log"]))


def init(prefix, levelname="INFO"):
    if os.getenv("DEBUG") == "True":
        print("Setting log level to DEBUG")
        levelname = "DEBUG"

    formatter = logging.Formatter(
        '%(asctime)s %(threadName)-10s %(name)s %(levelname)-8s %(message)s')
    LOGFILENAME = get_log_filepath(prefix)
    print("Redirecting log to", LOGFILENAME)
    handler = logging.FileHandler(filename=LOGFILENAME, mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(levels[levelname])
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(levels[levelname])
    logging.basicConfig(handlers=[handler, console], level=levelname)
