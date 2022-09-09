import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "config"
os.environ['OBJECT_CENTRIC_LIB_DATA'] = 'datasets'
DATA = Path(os.environ["OBJECT_CENTRIC_LIB_DATA"])
DEFAULT_VARIANTS_PATH = CONFIG / "dataset" / "variants" / "variants.yaml"
