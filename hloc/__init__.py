import logging
import sys

__version__ = '1.2'

logging.basicConfig(stream=sys.stdout,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
