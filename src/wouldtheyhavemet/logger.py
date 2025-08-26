import logging
import sys

# Configure a global logger for the package
logger = logging.getLogger("wouldtheyhavemet")
logger.setLevel(logging.INFO)

# Handler → console
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)