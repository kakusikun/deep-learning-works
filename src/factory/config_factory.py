import os
import os.path as osp
import sys
import datetime
import shutil

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.APEX = False
_C.OUTPUT_DIR = ""
_C.RESUME = ""
_C.EVALUATE = False
_C.SAVE = True
_C.IO = True
_C.SEED = 42
_C.EXPERIMENT = ""
_C.ENGINE = ""
_C.GRAPH = ""
_C.TRAINER = ""
_C.NUM_WORKERS = 16
_C.ORACLE = False
_C.EVALUATE_FREQ = 1

# -----------------------------------------------------------------------------
# Model in Graph
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.GPU = []
_C.MODEL.BACKBONE = ""
_C.MODEL.SAVE_CRITERION = "acc"
_C.MODEL.STRIDE = 1
_C.MODEL.FEATSIZE = 0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.PAD = 0
_C.INPUT.SIZE = (0, 0)
_C.INPUT.TRAIN_BS = 32
_C.INPUT.TEST_BS = 32
_C.INPUT.MEAN = []
_C.INPUT.STD = []
_C.INPUT.RAND_AUG_N = 2
_C.INPUT.RAND_AUG_M = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DB = CN()
_C.DB.PATH = ""
_C.DB.DATA = ""
_C.DB.DATA_FORMAT = ""
_C.DB.TARGET_FORMAT = ""
_C.DB.LOADER = ""
_C.DB.USE_TRAIN = True
_C.DB.USE_TEST = True
_C.DB.TRAIN_TRANSFORM = ""
_C.DB.TEST_TRANSFORM = ""
_C.DB.NUM_CLASSES = 0
_C.DB.NUM_KEYPOINTS = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "SGD"
_C.SOLVER.START_EPOCH = 1
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.CUSTOM = [] 
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS_FACTOR = 1.0
_C.SOLVER.NESTEROV = False
_C.SOLVER.AMSGRAD = False
_C.SOLVER.ITERATIONS_PER_EPOCH = 0
_C.SOLVER.LR_POLICY = ""

# for plateau
_C.SOLVER.MIN_LR = 0.0
_C.SOLVER.PLATEAU_SIZE = 10.0
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.MODEL_FREEZE_PEROID = 0

# for SWAG
_C.SOLVER.SWAG_RANK = 0
_C.SOLVER.SWAG_EPOCH_TO_COLLECT = 0
_C.SOLVER.SWAG_COLLECT_FREQ = 0

# -----------------------------------------------------------------------------
# FACEID
# -----------------------------------------------------------------------------
_C.FACEID = CN()
_C.FACEID.SIZE_PROBE = 5
_C.FACEID.LFW_PAIRSFILE_PATH = ""
_C.FACEID.PROBE_PATH = ""
_C.FACEID.GALLERY_PATH = ""
_C.FACEID.PROBE_TYPE = ""
_C.FACEID.GALLERY_TYPE = ""

# -----------------------------------------------------------------------------
# REID
# -----------------------------------------------------------------------------
_C.REID = CN()
_C.REID.SIZE_PERSON = 4
_C.REID.MSMT_ALL = False
_C.REID.CENTER_LOSS_LR = 0.5
_C.REID.CENTER_LOSS_WEIGHT = 0.0005

# -----------------------------------------------------------------------------
# Pedestrian Attribute Recognition
# -----------------------------------------------------------------------------
_C.PAR = CN()
_C.PAR.SELECT_CAT = -1
_C.PAR.IGNORE_CAT = []

# -----------------------------------------------------------------------------
# Pedestrian Attribute Recognition
# -----------------------------------------------------------------------------
_C.COCO = CN()
_C.COCO.TARGET = 'original'

# ---------------------------------------------------------------------------- #
# SPOS
# ---------------------------------------------------------------------------- #
_C.SPOS = CN()
_C.SPOS.EPOCH_TO_SEARCH = 60
_C.SPOS.CANDIDATE_RELAX_EPOCHS = 10
_C.SPOS.DURATION = 4

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #


def build_output(cfg, config_file=""):
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if cfg.EVALUATE:
        root = osp.join("evaluation", cfg.DB.DATA, cfg.EXPERIMENT)
    else:
        root = osp.join(os.getcwd(), "result", cfg.DB.DATA, cfg.EXPERIMENT)
    if not osp.exists(root):
        os.makedirs(root)
    n_folders = len([f for f in os.scandir(root) if f.is_dir()])
    cfg.OUTPUT_DIR = osp.join(root, f"{n_folders:03}-{time}")
    if cfg.OUTPUT_DIR and not osp.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
        if config_file != "":
            shutil.copy(config_file, osp.join(cfg.OUTPUT_DIR, config_file.split("/")[-1]))
    
def show_products():
    from src.factory.data_factory import DataFactory
    from src.factory.data_format_factory import DataFormatFactory
    from src.factory.transform_factory import TransformFactory
    from src.factory.loader_factory import LoaderFactory
    from src.factory.backbone_factory import BackboneFactory
    from src.factory.graph_factory import GraphFactory
    from src.factory.engine_factory import EngineFactory
    from src.factory.trainer_factory import TrainerFactory
    print("       DATA: ", DataFactory.get_products())
    print("DATA_FORMAT: ", DataFormatFactory.get_products())
    print("  TRANSFORM: ", TransformFactory.get_products())
    print("     LOADER: ", LoaderFactory.get_products())
    print("   BACKBONE: ", BackboneFactory.get_products())
    print("      GRAPH: ", GraphFactory.get_products())
    print("     ENGINE: ", EngineFactory.get_products())
    print("    TRAINER: ", TrainerFactory.get_products())
    sys.exit(1)

def show_configs():
    for c in _C:
        print("="*50)
        if isinstance(_C[c], CN):
            print(c)
            for cc in list(_C[c].keys()):
                print("-"*50)
                _type = str(type(_C[c][cc])).split(" ")[-1].split(">")[0].replace("\'", "")
                print(f"{cc:>30}    {_type:>6}    {_C[c][cc]}")
        else:
            _type = str(type(_C[c])).split(" ")[-1].split(">")[0].replace("\'", "")
            print(f"{c:<30}    {_type:>6}    {_C[c]}")
    print("="*50)
    sys.exit(1)