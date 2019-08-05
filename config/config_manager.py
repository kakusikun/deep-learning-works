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

_C.EXPERIMENT = ""

_C.TASK = ""

_C.MODEL = CN()
_C.MODEL.NUM_GPUS = 0
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.NAME = ""
_C.MODEL.PRETRAIN = "imagenet"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

_C.INPUT.IMAGE_PAD = 0
_C.INPUT.IMAGE_SIZE = (0, 0)
_C.INPUT.IMAGE_CROP_SIZE = (0, 0)
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 32
# Size of the image during test
_C.INPUT.SIZE_TEST = 32
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = []
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = []

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
_C.REID.SIZE_PERSON = 3

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = ""
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASET.TRAIN = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASET.TEST = ""

_C.DATASET.TRAIN_PATH = ""
_C.DATASET.TEST_PATH = ""
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.TRANSFORM = CN()
_C.TRANSFORM.HFLIP = False
_C.TRANSFORM.RANDOMCROP = False
_C.TRANSFORM.NORMALIZE = False
_C.TRANSFORM.RESIZE = False
_C.TRANSFORM.COLORJIT = False
_C.TRANSFORM.RANDOMERASING = False
_C.TRANSFORM.SINGLE_CROP = False
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.START_EPOCH = 0
_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.CENTER_LOSS_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS_FACTOR = 1.0

_C.SOLVER.NESTEROV = False
_C.SOLVER.WD_NORMALIZED = False

_C.SOLVER.WARMRESTART_MULTIPLIER = 2
_C.SOLVER.WARMRESTART_PERIOD = 10
_C.SOLVER.ITERATIONS_PER_EPOCH = -1
_C.SOLVER.LR_POLICY = "plateau"
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

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./result"
_C.RESUME = ""
_C.EVALUATE = ""
