import logging

__version__ = '1.2'

formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("hloc")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
