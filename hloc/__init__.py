import logging
from packaging import version

__version__ = '1.5'

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

try:
    import pycolmap
except ImportError:
    logger.warning('pycolmap is not installed, some features may not work.')
else:
    minimal_version = version.parse('0.3.0')
    found_version = pycolmap.__version__
    if found_version != 'dev':
        if version.parse(found_version) < minimal_version:
            logger.warning(
                'hloc now requires pycolmap>=%s but found pycolmap==%s, '
                'please upgrade with `pip install --upgrade pycolmap`',
                minimal_version, found_version)
