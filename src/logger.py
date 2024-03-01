import logging
import logging.handlers
import os

levels = {"INFO": logging.INFO,
          "WARN": logging.WARN,
          "DEBUG": logging.DEBUG,
          "ERROR": logging.ERROR}


def init(LOGFILENAME, levelname="INFO"):
    if os.getenv("DEBUG") == "True":
        print("Setting log level to DEBUG")
        levelname = "DEBUG"

    formatter = logging.Formatter(
        '%(asctime)s %(threadName)-10s %(name)s %(levelname)-8s %(message)s')
    handler = logging.FileHandler(filename=LOGFILENAME, mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(levels[levelname])
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(levels[levelname])
    logging.basicConfig(handlers=[handler, console], level=levelname)
