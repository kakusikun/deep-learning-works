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
_C.SOLVER.CENTER_LOSS_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS_FACTOR = 1.0
_C.SOLVER.NESTEROV = False

_C.SOLVER.LR_POLICY = ""

# for plateau
_C.SOLVER.MIN_LR = 0.0
_C.SOLVER.PLATEAU_SIZE = 10.0
_C.SOLVER.GAMMA = 0.1

# for cosine
_C.SOLVER.NUM_RESTART = 4
_C.SOLVER.T_MULT = 2
_C.SOLVER.T_0 = 10
_C.SOLVER.WD_NORMALIZED = False
_C.SOLVER.ITERATIONS_PER_EPOCH = -1

# for warmup
_C.SOLVER.WARMUP = False 
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_SIZE = 10.0

_C.SOLVER.MODEL_FREEZE_PEROID = 0

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
_C.REID.CYCLE = 30
_C.REID.MERGE = False
_C.REID.TRT = ""

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



def build_output(cfg, config_file=""):
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.EVALUATE:
        cfg.OUTPUT_DIR = osp.join("evaluation", cfg.DB.DATA, cfg.EXPERIMENT, time)
    else:
        cfg.OUTPUT_DIR = osp.join(os.getcwd(), "result", cfg.DB.DATA, cfg.EXPERIMENT, time)
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
    print("  TRANSFORM: ", TrainerFactory.get_products())
    print("     LOADER: ", LoaderFactory.get_products())
    print("   BACKBONE: ", BackboneFactory.get_products())
    print("      GRAPH: ", GraphFactory.get_products())
    print("     ENGINE: ", EngineFactory.get_products())
    print("    TRAINER: ", TrainerFactory.get_products())
    sys.exit(1)    