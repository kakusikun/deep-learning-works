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

cfg = CN()

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
cfg.APEX = False
cfg.OUTPUT_DIR = ""
cfg.RESUME = ""
cfg.EVALUATE = False
cfg.SAVE = True
cfg.IO = True
cfg.SEED = 42
cfg.EXPERIMENT = ""
cfg.ENGINE = ""
cfg.GRAPH = ""
cfg.TRAINER = ""
cfg.NUM_WORKERS = 16
cfg.ORACLE = False
cfg.EVALUATE_FREQ = 1
cfg.DISTRIBUTED = False
# -----------------------------------------------------------------------------
# Model in Graph
# -----------------------------------------------------------------------------
cfg.MODEL = CN()
cfg.MODEL.GPU = []
cfg.MODEL.BACKBONE = ""
cfg.MODEL.SAVE_CRITERION = "acc"
cfg.MODEL.STRIDES = [1]
cfg.MODEL.FEATSIZE = 0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
cfg.INPUT.PAD = 0
cfg.INPUT.SIZE = (0, 0)
cfg.INPUT.TRAIN_BS = 32
cfg.INPUT.TEST_BS = 32
cfg.INPUT.MEAN = []
cfg.INPUT.STD = []
cfg.INPUT.RAND_AUG_N = 2
cfg.INPUT.RAND_AUG_M = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DB = CN()
cfg.DB.PATH = ""
cfg.DB.DATA = ""
cfg.DB.DATA_FORMAT = ""
cfg.DB.TARGET_FORMAT = ""
cfg.DB.LOADER = ""
cfg.DB.USE_TRAIN = True
cfg.DB.USE_TEST = True
cfg.DB.TRAIN_TRANSFORM = ""
cfg.DB.TEST_TRANSFORM = ""
cfg.DB.NUM_CLASSES = 0
cfg.DB.NUM_KEYPOINTS = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
cfg.SOLVER.OPTIMIZER = "SGD"
cfg.SOLVER.START_EPOCH = 1
cfg.SOLVER.MAX_EPOCHS = 50
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.BIAS_LR_FACTOR = 1.0
cfg.SOLVER.CUSTOM = [] 
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 0.0005
cfg.SOLVER.WEIGHT_DECAY_BIAS_FACTOR = 1.0
cfg.SOLVER.NESTEROV = False
cfg.SOLVER.AMSGRAD = False
cfg.SOLVER.ITERATIONS_PER_EPOCH = 0
cfg.SOLVER.LR_POLICY = ""

# for plateau
cfg.SOLVER.MIN_LR = 0.0
cfg.SOLVER.PLATEAU_SIZE = 10.0
cfg.SOLVER.GAMMA = 0.1

cfg.SOLVER.MODEL_FREEZE_PEROID = 0

# for SWAG
cfg.SOLVER.SWAG_RANK = 0
cfg.SOLVER.SWAG_EPOCH_TO_COLLECT = 0
cfg.SOLVER.SWAG_COLLECT_FREQ = 0

# -----------------------------------------------------------------------------
# FACEID
# -----------------------------------------------------------------------------
cfg.FACEID = CN()
cfg.FACEID.SIZE_PROBE = 5
cfg.FACEID.LFW_PAIRSFILE_PATH = ""
cfg.FACEID.PROBE_PATH = ""
cfg.FACEID.GALLERY_PATH = ""
cfg.FACEID.PROBE_TYPE = ""
cfg.FACEID.GALLERY_TYPE = ""

# -----------------------------------------------------------------------------
# REID
# -----------------------------------------------------------------------------
cfg.REID = CN()
cfg.REID.NUM_PERSON = 0
cfg.REID.SIZE_PERSON = 4
cfg.REID.MSMT_ALL = False
cfg.REID.CENTER_LOSS_LR = 0.5
cfg.REID.CENTER_LOSS_WEIGHT = 0.0005

# -----------------------------------------------------------------------------
# Pedestrian Attribute Recognition
# -----------------------------------------------------------------------------
cfg.PAR = CN()
cfg.PAR.SELECT_CAT = -1
cfg.PAR.IGNORE_CAT = []

# -----------------------------------------------------------------------------
# Pedestrian Attribute Recognition
# -----------------------------------------------------------------------------
cfg.COCO = CN()
cfg.COCO.TARGET = 'original'

# ---------------------------------------------------------------------------- #
# SPOS
# ---------------------------------------------------------------------------- #
cfg.SPOS = CN()
cfg.SPOS.EPOCH_TO_SEARCH = 60
cfg.SPOS.CANDIDATE_RELAX_EPOCHS = 10
cfg.SPOS.DURATION = 4

# ---------------------------------------------------------------------------- #
# YOLOv3                                     
# ---------------------------------------------------------------------------- #
cfg.YOLO = CN()
cfg.YOLO.ANCHORS = []


def build_output(cfg, config_file="", find_existing_path=False):
    if not find_existing_path:
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
    else:
        root = osp.join(os.getcwd(), "result", cfg.DB.DATA, cfg.EXPERIMENT)
        cfg.OUTPUT_DIR = osp.join(root, sorted(os.listdir(root))[-1])

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
    for c in cfg:
        print("="*50)
        if isinstance(cfg[c], CN):
            print(c)
            for cc in list(cfg[c].keys()):
                print("-"*50)
                _type = str(type(cfg[c][cc])).split(" ")[-1].split(">")[0].replace("\'", "")
                print(f"{cc:>30}    {_type:>6}    {cfg[c][cc]}")
        else:
            _type = str(type(cfg[c])).split(" ")[-1].split(">")[0].replace("\'", "")
            print(f"{c:<30}    {_type:>6}    {cfg[c]}")
    print("="*50)
    sys.exit(1)