import os
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

_A = CN()
_A.INPUT = CN()
_A.INPUT.TYPE = "image"
_A.INPUT.PATH = ""

_A.DNET = CN()
_A.DNET.TYPE = "gt"
_A.DNET.PATH = ""
_A.DNET.INPUT_SIZE = ()
_A.SAVE = False
_A.SAVE_OUTPUT = ""
_A.PNET = ""
_A.DATABASE = ""

_C = CN()

_C.EXPERIMENT = ""

_C.TASK = ""

_C.ENGINE = ""

_C.MANAGER = ""

_C.TRAINER = ""

_C.TRAIN_TRANSFORM = ""
_C.TEST_TRANSFORM = ""

_C.NUM_WORKERS = 16

_C.ORACLE = False

_C.MODEL = CN()
_C.MODEL.GPU = []

_C.MODEL.NAME = ""
_C.MODEL.PRETRAIN = "own"
_C.MODEL.TASK = "classifier"
_C.MODEL.POOLING = "AVG"
_C.MODEL.SAVE_CRITERION = "acc"
_C.MODEL.NORM = "BN"
_C.MODEL.OUTPUT_STRIDE = 32

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

_C.INPUT.PAD = 0
_C.INPUT.RESIZE = (0, 0)
_C.INPUT.CROP_SIZE = (0, 0)
# Size of the image during training
_C.INPUT.TRAIN_BS = 32
# Size of the image during test
_C.INPUT.TEST_BS = 32
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.MEAN = []
# Values to be used for image normalization
_C.INPUT.STD = []

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

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DB = CN()
_C.DB.NUM_CLASSES = 0
_C.DB.NUM_KEYPOINTS = 0
_C.DB.DATA = ""
_C.DB.DATASET = ""
_C.DB.LOADER = ""
_C.DB.USE_TRAIN = True
_C.DB.USE_TEST = True
_C.DB.ATTENTION_MAPS = ""
_C.DB.ATTENTION_MAPS_LIST = ""
_C.DB.PATH = ""

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.START_EPOCH = 0
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

# for cosine
_C.SOLVER.WARMRESTART_MULTIPLIER = 2
_C.SOLVER.WARMRESTART_PERIOD = 10
_C.SOLVER.WD_NORMALIZED = False
_C.SOLVER.ITERATIONS_PER_EPOCH = -1

_C.SOLVER.CYCLIC_MAX_LR = 1.0

_C.SOLVER.NUM_LOSSES = 0
_C.SOLVER.EVALUATE_FREQ = 1
_C.SOLVER.LOG_FREQ = 1

_C.SOLVER.LR_STEPS = []
_C.SOLVER.WARMUP = False 
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_SIZE = 10.0
_C.SOLVER.PLATEAU_SIZE = 10.0

_C.SOLVER.MODEL_FREEZE_PEROID = 0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.RESUME = ""
_C.EVALUATE = ""

def build_output(cfg, config_file=""):
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.EVALUATE:
        cfg.OUTPUT_DIR = os.path.join("evaluation", cfg.TASK, cfg.EXPERIMENT, time)
    else:
        cfg.OUTPUT_DIR = os.path.join("result", cfg.TASK, cfg.EXPERIMENT, time)
    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
        if config_file != "":
            shutil.copy(config_file, os.path.join(cfg.OUTPUT_DIR, config_file.split("/")[-1]))
