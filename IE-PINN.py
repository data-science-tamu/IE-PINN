# ================================= Imports ================================== #
import math
import warnings
import torch
import numpy as np
# from sklearn import preprocessing
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import torch
from itertools import product

from typing import Tuple

# Simulation instruction ------------------------------------------------------#
# 1) Select simulation type
'''
1) Elasnet from author code 
2) Proposed Separated NN 
3) Proposed Separated NN and 2 NN for Nu and E
4) Non-separated NN : Swish-Siren
5) Non-separated NN : Siren-Relu
6) Non-separated NN : Swish-Relu
7) Separated NN with original Activation function
8) Proposed Separated NN without pretraining
'''
SIMULATION_OPTION = 2

# 2) Select GPU number
CUDA_INDEX = 1

# 3) Data type
'''
1) General
2) SNR1000
3) SNR500
4) SNR100
5) Lowres
6) SNR in manuscript run
'''
DATA_TYPE = 5  #6

# 4) Define dataset and SNR to be included
IS_MANUSCRIPT_RUN = False

TESTRUN = True #False

FIT_STRAIN_HIGH_RES_CORD = True #False
EVALUATE_ELAS_HIGH_RES = False #True
# ---------------------------------------------------------------------------- #
#Visualization
CMAP = 'jet' 

BATCH_START_TIME = time.time()


# ========================= Program Setup / Settings ========================= #
# ------------------------------ DEBUG SETTINGS ------------------------------ #
STATE_MESSAGES = True
NOTIFY_ITERATION_MOD = 100

# ------------------------------- DEVICE SETUP ------------------------------- #
# DEFAULT_CPU = input("Default to CPU (y/n):").lower() == "y"
DEFAULT_CPU = False  # If false defaults to gpu
TENSOR_TYPE = torch.float32

# ------------------------------- EXPERIMENTAL SETUP ------------------------------- #
SNR = None
if IS_MANUSCRIPT_RUN:
    DEVELOPMENT_MODE = False #True
    INVESTIGATION_MODE = False #True
    SIMULATION_TYPE = "NOISY"
    if DATA_TYPE == 2:
        SNR = 1000
    elif DATA_TYPE == 3:
        SNR = 500
    elif DATA_TYPE == 4:
        SNR = 100
else:
    DEVELOPMENT_MODE = False #True
    INVESTIGATION_MODE = False #True
    if DATA_TYPE == 1:
        SIMULATION_TYPE = "GENERAL"
    elif DATA_TYPE == 5:
        SIMULATION_TYPE = "LOWRES"
    else:
        SIMULATION_TYPE = "NOISY"
        if DATA_TYPE == 2:
            SNR = 1000
        elif DATA_TYPE == 3:
            SNR = 500
        elif DATA_TYPE == 4:
            SNR = 100

# ------------------------------- MODEL SETUP ------------------------------- #

MODEL_NAME = "SIREN"             # "SIREN" or "SIRE" = SILU-RELU

FIT_DISPLACEMENT = True
FIT_STRAIN = True
SEPARATED_ENu = False
RETRAIN_DISPLACEMENT = True
RETRAIN_INTENSITY = True
WITH_PRETRAIN = True
IS_NU_SIGMOID = False #True

if SIMULATION_OPTION == 3:
    SEPARATED_ENu = True
if SIMULATION_OPTION == 7:
    FIT_STRAIN = False
if SIMULATION_OPTION == 8:
    WITH_PRETRAIN = False  # True

# ------------------------------- LOSS FUNCTION SETUP ------------------------------- #
DIFFERTIATION_METHOD = "FD"     # "FD" or "AD"
LOSS_NORM = "L1"                # "L1", "L2"
LOSS_MEAN_E = True

# ----------------------------- MECHANICAL ASSUMPTION -----------------------------#
# Focused parameters
IS_COMPRESSIBLE = True #False #True #False
PARAMETRIC_SET = 'A'  # A,B,C
ACTIVATION_I_SIGMOID = True #False


# ----------------------------- DATASET NAME -----------------------------#

DATAFOLDER = 'compressible'
DATASETNAME = "m_z5_nu_z1" # "m_z9_nu_z8"
PARAMETRIC_E = False
FIXED_NU = False


# ----------------------------- SCALING E CALCULATION -----------------------------#
SCALING_E = True


# ----------------------------- GPU/CPU SETTING  -----------------------------#
# Selecting device to use. If use_cpu is true, will default
# to cpu, otherwise will use GPU if a GPU is available
# NOTE: When DEVICE is changed, call device_refresh to move ELAS_OUTPUT_SHAPE
# to correct device
if DEFAULT_CPU:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device("cuda", index=CUDA_INDEX)

if STATE_MESSAGES: print("STATE: torch device is", DEVICE)

# ----------------------------- TRAINING SETTINGS ---------------------------- #
NUM_FITTING_1_EPOCHS = 50
NUM_FITTING_2_EPOCHS = 100
NUM_FITTING_3_EPOCHS = 50  # Intensity prediction
NUM_TRAINING_EPOCHS = 200  # Each epoch is 1000 training "sessions" (see below)
ITERATIONS_PER_EPOCH = 1000

if SIMULATION_OPTION == 8:
    NUM_TRAINING_EPOCHS = 350
# # -------------- TRIAL -------------------------
# NUM_FITTING_1_EPOCHS = 11
# NUM_FITTING_2_EPOCHS = 11
# NUM_TRAINING_EPOCHS = 11 # Each epoch is 1000 training "sessions" (see below)
# ITERATIONS_PER_EPOCH = 11
# PREFIX = "Trial"
# MODEL_NAME = "Trial"
# GROUP_TAG1="Trial"

# ------------------------------ DATA PARAMETERS ----------------------------- #
# Where to find data (assumes follow same naming scheme as paper)
## -------------- DEFAULT PARAMETERS -------------------------
PATH_TO_DATA = ".././data"
FILENAME_disp_coord = "disp_coord"
FILENAME_disp_data = "disp_data"
FILENAME_strain_coord = "strain_coord"
FILENAME_elas_coord_array = "strain_coord"
FILENAME_strain_data = "strain_data"
GROUP_TAG1 = None
GROUP_TAG2 = None
GROUP_TAG3 = None
GROUP_TAG4 = None
GROUP_TAG5 = None
DOWNSAMPLING_N = None

if IS_MANUSCRIPT_RUN:
    # #  NOISY DATA
    TRIAL_NAME = DATASETNAME
    PREFIX = f"SIREN_Noise_SNR_{SNR}"
    GROUP_TAG1 = f"MNS-SNR{SNR}"
    REGULARIZATION_L = 'NONE'
else:
    if SIMULATION_TYPE == "GENERAL":
        TRIAL_NAME = DATASETNAME
        # #  GENERAL DATA
        # -------------------------------------------------------
        GROUP_TAG1 = "General"
        REGULARIZATION_L = 'NONE'
        PREFIX = f"SIREN_GENERAL"
        # -------------------------------------------------------
    elif SIMULATION_TYPE == "NOISY":
        if DATASETNAME == "m_z5_nu_z11":
            # #  NOISY DATA
            # -------------------------------------------------------
            TRIAL_NAME = f"{DATASETNAME}_noisy"
            FILENAME_disp_data= f"disp_data_SNR_{SNR}"
        else:
            # #  NOISY DATA
            TRIAL_NAME = f"{DATASETNAME}"
        PREFIX = f"SIREN_Noise_SNR_{SNR}"
        GROUP_TAG1 = "Noisy"
        REGULARIZATION_L = 'NONE' #'L1' # 0,1,NONE
        # -------------------------------------------------------


    # -------------------------------------------------------


# # No Fitting Displacement
# FIT_DISPLACEMENT = False
# PREFIX += "_No_Disp_Fit"


# Output shape, assuming that displacement shape is 1 larger
# If data shape is (256, 256) then displacement is assumed at (257, 257)

if DATASETNAME[2] == "z" or DATASETNAME == 'm_TSN_nu_TML' or DATASETNAME == 'm_TML_nu_TSN':
    ELAS_INPUT_SHAPE = torch.tensor([256, 256], device=DEVICE)
    DISP_INPUT_SHAPE = torch.tensor([257, 257], device=DEVICE)
    DISP_OUTPUT_SHAPE = torch.tensor([257, 257], device=DEVICE)
    if SIMULATION_TYPE == "LOWRES" and not FIT_STRAIN_HIGH_RES_CORD:
        STRAIN_INPUT_SHAPE = torch.tensor([129, 129], device=DEVICE)
        STRAIN_OUTPUT_SHAPE = torch.tensor([128, 128], device=DEVICE)
        ELAS_OUTPUT_SHAPE = torch.tensor([128, 128], device=DEVICE)
    else:
        STRAIN_INPUT_SHAPE = torch.tensor([257, 257], device=DEVICE)
        STRAIN_OUTPUT_SHAPE = torch.tensor([256, 256], device=DEVICE)
        ELAS_OUTPUT_SHAPE = torch.tensor([256, 256], device=DEVICE)
    NUM_ROWS_ELAS = 256
    NUM_COLS_ELAS = 256
    NUM_ROWS_DISP = 257
    NUM_COLS_DISP = 257
    IS_DIM_PREDEFINED = True
else:
    with open(f"{PATH_TO_DATA}/{DATAFOLDER}/{DATASETNAME}/data_description.txt", "r") as file:
        lines = file.readlines()
    second_line = lines[1].strip()
    if "is" in second_line:
        STRING_DIMENSION = second_line.split("is", 1)[1].strip()
        NUM_COLS_ALL, NUM_ROWS_ALL = map(int, STRING_DIMENSION.strip("()").split(","))
    else:
        NUM_COLS_ALL = 256
        NUM_ROWS_ALL = 256

    ELAS_INPUT_SHAPE = torch.tensor([NUM_ROWS_ALL-1, NUM_COLS_ALL-1], device=DEVICE)
    DISP_INPUT_SHAPE = torch.tensor([NUM_ROWS_ALL, NUM_COLS_ALL], device=DEVICE)
    DISP_OUTPUT_SHAPE = torch.tensor([NUM_ROWS_ALL, NUM_COLS_ALL], device=DEVICE)
    STRAIN_INPUT_SHAPE = torch.tensor([NUM_ROWS_ALL, NUM_COLS_ALL], device=DEVICE)
    ELAS_OUTPUT_SHAPE = torch.tensor([NUM_ROWS_ALL-1, NUM_COLS_ALL-1], device=DEVICE)
    STRAIN_OUTPUT_SHAPE = torch.tensor([NUM_ROWS_ALL-1, NUM_COLS_ALL-1], device=DEVICE)
    NUM_ROWS_ELAS = NUM_ROWS_ALL-1
    NUM_COLS_ELAS = NUM_COLS_ALL-1
    NUM_ROWS_DISP = NUM_ROWS_ALL
    NUM_COLS_DISP = NUM_COLS_ALL
    IS_DIM_PREDEFINED = False


STRAIN_SIZE = 3  # [ε_xx, ε_yy, γ_xy] (epsilon/e, epsilon/e, gamma/r) # Strain Fit out
DISPLACEMENT_SIZE = 2  # [u_x, u_y] # DispFit out
INTENSITY_SIZE = 1 # [I] # Intensity prediction
COORDINATE_SIZE = 2  # [x, y] # Model in
if PARAMETRIC_E or SEPARATED_ENu:
    ELAS_SIZE = 1  # [ pred_E or pred_v] # Elas out
else:
    ELAS_SIZE = 2  # [pred_E, pred_v] # Elas out


# ----------------------------- MODEL PARAMETERS ----------------------------- #
LEARN_RATE = 0.001
# TODO: ask Dr. about using dropout layers.
# https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9
# https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/
# Mentions a p of 0.2 is a good starting point

NUM_NEURON_FIT = 128  # MUST BE EVEN (depth = NUM_NEURON // 2)
D_POS_ENC_FIT = NUM_NEURON_FIT // 2  # depth
assert (NUM_NEURON_FIT % 2 == 0)  # NOTE: ONLY FOR 2D INPUT DATA


# TEMPORARY for testing
class SirenAct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


NUM_HIDDEN_LAYER_FIT = 16
ACTIVATION_FUNCTION_FIT = SirenAct  # torch.nn.SiLU # aka "swish", the paper said it was best for displacement
ACTIVATION_FUNCTION_OUT_FIT = None  # lambda x : x #torch.nn.SiLU

ACT_DISP = ACTIVATION_FUNCTION_FIT
ACT_DISP_OUT = ACTIVATION_FUNCTION_OUT_FIT
ACT_STRAIN = ACTIVATION_FUNCTION_FIT
ACT_STRAIN_OUT = ACTIVATION_FUNCTION_OUT_FIT
ACT_INTENSITY = ACTIVATION_FUNCTION_FIT
if ACTIVATION_I_SIGMOID:
    ACT_INTENSITY_OUT = torch.nn.Sigmoid
else:
    ACT_INTENSITY_OUT = torch.nn.Hardsigmoid

NUM_NEURON_ELAS = 128
NUM_HIDDEN_LAYER_ELAS = 16
ACTIVATION_FUNCTION_ELAS = SirenAct  # torch.nn.ReLU
ACTIVATION_FUNCTION_OUT_ELAS = torch.nn.Softplus


if SIMULATION_OPTION == 4:
    ACTIVATION_FUNCTION_FIT = torch.nn.SiLU # aka "swish", the paper said it was best for displacement
    GROUP_TAG3 = f"SILU_SIREN"
elif SIMULATION_OPTION == 5:
    ACTIVATION_FUNCTION_ELAS = torch.nn.ReLU
    GROUP_TAG3 = f"SIREN_RELU"
elif SIMULATION_OPTION == 6 or SIMULATION_OPTION == 7:
    ACTIVATION_FUNCTION_FIT = torch.nn.SiLU # aka "swish", the paper said it was best for displacement
    ACTIVATION_FUNCTION_ELAS = torch.nn.ReLU
    GROUP_TAG3 = f"SILU_RELU"
else:
    GROUP_TAG3 = f"SIREN_SIREN"


# ------------------------------ LOSS PARAMETERS ----------------------------- #
WEIGHT_D_DISP = 2.0
WEIGHT_D_STRAIN = 1.0
WEIGHT_INT = 2.0
if LOSS_MEAN_E:
    WEIGHT_E = 0.02
else:
    WEIGHT_E = 0.00
WEIGHT_R = 3.0  # Equilibrium condition Loss. Loss_r
WEIGHT_L = 0.1 # Regularization weight


if SIMULATION_TYPE == "LOWRES":
    WEIGHT_D_DISP = 10.0
    WEIGHT_D_STRAIN = 5.0
    NUM_FITTING_2_EPOCHS = 50


if TESTRUN:
    NUM_FITTING_1_EPOCHS = 1
    NUM_FITTING_2_EPOCHS = 1

if SIMULATION_TYPE == "NOISY":
    if REGULARIZATION_L == 'NONE':
        GROUP_TAG3 = f"{REGULARIZATION_L} REG"
    else:
        GROUP_TAG3 = f"{REGULARIZATION_L}-{WEIGHT_L}"

E_CONSTRAINT = 0.25

if E_CONSTRAINT == 0.5:
    WEIGHT_D_DISP = 4.0
    WEIGHT_D_STRAIN = 2.0

GROUP_TAG4 = f"{DIFFERTIATION_METHOD}"
GROUP_TAG5 = f"LOSSNORM-{LOSS_NORM}"

# ------------------------------ OUTPUT FOLDERS ------------------------------ #
# Output Path. NOTE: Must already be present in file system relative to where
# script is ran.
OUTPUT_FOLDER = ".././results"
OUTPUT_FOLDER_MODEL = OUTPUT_FOLDER
MODEL_SUBFOLDER = "/models"
PRE_FIT_MODEL_SUBFOLDER = "/pre_fitted"
# LOSS_SUBFOLDER = "/loss" # TODO: Implement this
SAVE_FIT2_LOSS = True  # FIT2: [i_net, ]
SAVE_TRAIN_LOSS = True  # TRAIN: [i_net, wld_d, wld_s, wle, wlr, total], # Weighted.

SAVES_PER_TRAIN = 10
SAVE_INTERVAL_TRAIN = ITERATIONS_PER_EPOCH // SAVES_PER_TRAIN
SAVES_PER_FIT2 = 10
SAVE_INTERVAL_LOSS_F2 = ITERATIONS_PER_EPOCH // SAVES_PER_FIT2


# -------------------------- Parameter Verification -------------------------- #
def refresh_devices():
    ELAS_OUTPUT_SHAPE.to(DEVICE)


# ------------------------- Logging Parameters ------------------------------- #
def logging_parameters(Folderlocation):
    # Capture all global variables dynamically
    parameters = {
        key: value for key, value in globals().items()
        if not key.startswith("_") and not isinstance(value, (list, dict, type(math))) and not callable(value)
    }
    # Save to a text file
    makeDirIfNotExist(f"{Folderlocation}/parameters.txt")
    with open(f"{Folderlocation}/parameters.txt", "w") as f:
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")

# ---------------------------------------------------------------------------- #
# ----------------------------- NEPTUNE SETTINGS  ---------------------------- #
# PREFIX = "TestTrial"
OUTPUT_FOLDER = OUTPUT_FOLDER + f"/{PREFIX}"
if FIT_DISPLACEMENT:
    NAMES_LOSS = ["loss_d_disp", "loss_d_strain", "loss_E", "loss_r", "total_loss"]
    TAGS_LOSS = [
        "evaluation/loss_d_disp",
        "evaluation/loss_d_strain",
        "evaluation/loss_E",
        "evaluation/loss_r",
        "evaluation/total_loss"
    ]
else:
    NAMES_LOSS = ["loss_E", "loss_r", "total_loss"]
    TAGS_LOSS = [
        "evaluation/loss_E",
        "evaluation/loss_r",
        "evaluation/total_loss"
    ]

import neptune
from neptune.exceptions import NeptuneException


class Neptune:
    ACTION_SET = 1
    ACTION_ADD = 2
    ACTION_APPEND = 3
    ACTION_UPLOAD = 4

    def __init__(self, project = "xx",
                 api_token = "xx"):
        self.run = neptune.init_run(
            project=project,
            api_token=api_token,
        )  # your credentials
        self.add_tags(PREFIX)
        self.add_group_tags(MODEL_NAME)
        # RUN["sys/tags"].add(PREFIX)
        # RUN["sys/group_tags"].add(MODEL_NAME)

        if GROUP_TAG1 is not None:
            self.add_group_tags(GROUP_TAG1)
            # RUN["sys/group_tags"].add(GROUP_TAG1)
            # RUN["sys/group_tags"].add("sig1_02")
            # RUN["sys/group_tags"].add("sig2_004")

        if GROUP_TAG2 is not None:
            self.add_group_tags(GROUP_TAG2)

        if GROUP_TAG3 is not None:
            self.add_group_tags(GROUP_TAG3)
        if GROUP_TAG4 is not None:
            self.add_group_tags(GROUP_TAG4)
        if GROUP_TAG5 is not None:
            self.add_group_tags(GROUP_TAG5)

        # RUN["sys/group_tags"].add(GROUP_TAG2)
        # RUN["sys/description"].add(f"{GROUP_TAG1}")

        if SNR is not None:
            self.set_value_to_field("setting/SNR", SNR)
            # RUN["setting/SNR"] = SNR
        if DOWNSAMPLING_N is not None:
            self.set_value_to_field("setting/DOWNSAMPLING_N", DOWNSAMPLING_N)
            # RUN["setting/DOWNSAMPLING_N"] = DOWNSAMPLING_N

        self.set_value_to_field("setting/n_epochs", NUM_TRAINING_EPOCHS)
        self.set_value_to_field("setting/n_per_epoch", ITERATIONS_PER_EPOCH)
        self.set_value_to_field("setting/prefix", PREFIX)
        self.set_value_to_field("setting/filename_disp_data", FILENAME_disp_data)
        self.set_value_to_field("setting/filename_disp_coord", FILENAME_disp_coord)
        #         RUN["setting/n_epochs"]= NUM_TRAINING_EPOCHS
        #         RUN["setting/n_per_epoch"]=ITERATIONS_PER_EPOCH
        #         RUN["setting/prefix"] = PREFIX
        #         RUN["setting/filename_disp_data"] = FILENAME_disp_data
        #         RUN["setting/filename_disp_coord"] = FILENAME_disp_coord

        # params = {"Run_Name": PREFIX, "PINN/W_disp": WEIGHT_D_DISP, "PINN/W_Strain": WEIGHT_D_STRAIN, "PINN/W_E": WEIGHT_E, "PINN/W_PDE":WEIGHT_R, "PINN/E_CONSTANT": E_CONSTRAINT, "NN/Width":NUM_NEURON_ELAS, "NN/Depth":NUM_HIDDEN_LAYER_ELAS, "NN/Activation(HiddenLayer,u)": ACT_DISP, "NN/Activation(OutputLayer,u)": ACT_DISP_OUT, "NN/Activation(HiddenLayer,E)": ACTIVATION_FUNCTION_ELAS, "NN/Activation(OutputLayer,E)": ACTIVATION_FUNCTION_OUT_ELAS, "Optimization/NumIterPerEpoch": ITERATIONS_PER_EPOCH, "Optimization/NumEpochs": NUM_TRAINING_EPOCHS}
        params = {
            "Run_Name": PREFIX,
            "PINN/W_disp": WEIGHT_D_DISP,
            "PINN/W_Strain": WEIGHT_D_STRAIN,
            "PINN/W_E": WEIGHT_E,
            "PINN/W_PDE": WEIGHT_R,
            "PINN/E_CONSTANT": E_CONSTRAINT,
            "NN/Width": NUM_NEURON_ELAS,
            "NN/Depth": NUM_HIDDEN_LAYER_ELAS,
            "NN/Activation(HiddenLayer,u)": str(ACT_DISP),  # Convert to string
            "NN/Activation(OutputLayer,u)": str(ACT_DISP_OUT),  # Convert to string
            "NN/Activation(HiddenLayer,E)": str(ACTIVATION_FUNCTION_ELAS),  # Convert to string
            "NN/Activation(OutputLayer,E)": str(ACTIVATION_FUNCTION_OUT_ELAS),  # Convert to string
            "Optimization/NumIterPerEpoch": ITERATIONS_PER_EPOCH,
            "Optimization/NumEpochs": NUM_TRAINING_EPOCHS,
        }
        self.set_value_to_field("parameters", params)
        # RUN["parameters"] = params

    # Function to safely log data to Neptune
    def safe_log(self, key, action, value=None):
        """
        Safely performs logging to Neptune, catching BrokenPipeError and other exceptions.

        Parameters:
        - run: Neptune run object.
        - key: The field to log to in Neptune.
        - action: The logging action ('set', 'add', 'append').
        - value: The value to log or increment (optional, depends on action).
        """
        try:
            if action == Neptune.ACTION_SET:
                self.run[key] = value  # Set value
            elif action == Neptune.ACTION_ADD:
                self.run[key].add(value)  # Increment value
            elif action == Neptune.ACTION_APPEND:
                self.run[key].append(value)  # Append value
            elif action == Neptune.ACTION_UPLOAD:
                self.run[key].upload(value)  # Append value
            else:
                raise ValueError(f"Unsupported action '{action}'. Use 'set', 'add', or 'append'.")
        except BrokenPipeError:
            warnings.warn(f"BrokenPipeError encountered while logging {key}. Retrying or skipping.")
            # Retry logic or alternative actions can be added here
        except NeptuneException as e:
            warnings.warn(f"NeptuneException encountered: {e}. Please check Neptune configuration.")
        except Exception as e:
            warnings.warn(f"Unexpected error while logging {key}: {e}")

    def add_tags(self, value):
        self.safe_log("sys/tags", Neptune.ACTION_ADD, value)
        # self.run["sys/tags"].add(value)

    def add_group_tags(self, value):
        self.safe_log("sys/group_tags", Neptune.ACTION_ADD, value)
        # self.run["sys/group_tags"].add(value)

    def add_value_to_field(self, field, value):
        self.safe_log(field, Neptune.ACTION_ADD, value)
        # self.run[field].add(value)

    def set_value_to_field(self, field, value):
        self.safe_log(field, Neptune.ACTION_SET, value)
        # self.run[field] = value

    def append_value_to_field(self, field, value):
        # self.run[field].append(value)
        self.safe_log(field, Neptune.ACTION_APPEND, value)

    def upload_to_field(self, field, value):
        self.safe_log(field, Neptune.ACTION_UPLOAD, value)
        # self.run[field].upload(value)


# ============================= Residual Block ============================= #
class ResidualBlock(torch.nn.Module):
    def __init__(self, features, activation=torch.nn.ReLU):
        super().__init__()
        self.linear1 = torch.nn.Linear(features, features)
        self.activation1 = activation()
        self.linear2 = torch.nn.Linear(features, features)
        self.activation2 = activation()

    def forward(self, x):
        identity = x  # Keep the input as identity
        out = self.activation1(self.linear1(x))  # First transformation
        out = self.linear2(out)  # Second transformation (no activation here)
        out += identity  # Add the input to the transformed output
        return self.activation2(out)  # Apply activation after addition
    # ---------------------------------------------------------------------------- #


# ========================= Positional Encoding Model ======================== #
class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, depth: int, min_freq=1e-4):
        super().__init__()
        self.dim_in = 2
        self.dim_out = depth * 2

        depth = torch.tensor(depth, device=DEVICE)
        min_freq = torch.tensor(min_freq, device=DEVICE)
        self.freq_tensor = min_freq ** (
                2 * (
                torch.arange(depth, device=DEVICE) // 2
        ) / depth
        )

    # x = pos_tensor = A_nxm, a_ij =[i,j]
    def forward(self, coord: torch.Tensor):
        # NOTE: In place operations here make computing gradients difficult
        pos_x = coord[:, 0]
        pos_enc_x = pos_x.reshape(-1, 1) * self.freq_tensor.reshape(1, -1)
        # pos_enc_x[:, ::2] = torch.cos(pos_enc_x[:, ::2])
        # pos_enc_x[:, 1::2] = torch.sin(pos_enc_x[:, 1::2])
        pos_enc_x_cos = torch.cos(pos_enc_x[:, ::2])
        pos_enc_x_sin = torch.sin(pos_enc_x[:, 1::2])
        pos_enc_x = torch.concat([pos_enc_x_cos, pos_enc_x_sin], dim=1)

        pos_y = coord[:, 1]
        pos_enc_y = pos_y.reshape(-1, 1) * self.freq_tensor.reshape(1, -1)
        # pos_enc_y[:, ::2] = torch.cos(pos_enc_y[:, ::2] )
        # pos_enc_y[:, 1::2] = torch.sin(pos_enc_y[:, 1::2])
        pos_enc_y_cos = torch.cos(pos_enc_y[:, ::2])
        pos_enc_y_sin = torch.sin(pos_enc_y[:, 1::2])
        pos_enc_y = torch.concat([pos_enc_y_cos, pos_enc_y_sin], dim=1)

        return torch.cat([pos_enc_x, pos_enc_y], dim=1)


# ---------------------------------------------------------------------------- #

# ===================== Fitting Model and Loss Component ===================== #
class FittingModel(torch.nn.Module):
    # Model Structure, parent of the different types of fitted inputs
    def __init__(self, in_num, out_num, act_hidden=ACTIVATION_FUNCTION_FIT, act_out=ACTIVATION_FUNCTION_OUT_FIT):
        super().__init__()

        self.num_layers = NUM_HIDDEN_LAYER_FIT

        self.pos_encode = PositionalEncoding2D(D_POS_ENC_FIT)

        self.hidden1 = torch.nn.Linear(self.pos_encode.dim_out, NUM_NEURON_FIT)
        for i in range(2, self.num_layers):
            setattr(self, f"hidden{i}", torch.nn.Linear(NUM_NEURON_FIT, NUM_NEURON_FIT))
        self.out = torch.nn.Linear(NUM_NEURON_FIT, out_num)

        self.act1 = act_hidden()
        for i in range(2, self.num_layers):
            setattr(self, f"act{i}", act_hidden())
        # self.act_out = act_out()

    def forward(self, x: torch.Tensor):
        x = self.pos_encode(x)
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")(getattr(self, f"hidden{i}")(x))

        # x = self.pos_encode(x)
        #
        # x_0 = 0
        # for i in range(1, self.num_layers-1, 2):
        #     x = getattr(self, f"hidden{i}")(x) # linear 1
        #     x = getattr(self, f"act{i}")(x) # act 1
        #     x = getattr(self, f"hidden{i+1}")(x) # linear 2
        #     x = getattr(self, f"act{i+1}")(x + x_0) # act 2 (x + prev_x)
        #     x_0 = x
        # x = self.act15(self.hidden15(x))
        x = self.out(x)
        # x = self.act_out(x)
        return x

    def init_weight_and_bias(layer):
        std_dev = 0.1
        max_dev = 2  # Maximum number of standard deviations from mean
        mean = 0
        initial_bias = 0.1
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.trunc_normal_(
                layer.weight,
                mean=mean,
                std=std_dev,
                a=(-max_dev * std_dev),
                b=(max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)

    def create_coordinate_tensor(self, coordinates: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            coordinates.reshape(-1, 2),
            dtype=TENSOR_TYPE,
            device=DEVICE,
            # requires_grad=True # TODO see if this is needed (later)
        )

    def save_eval(self, coordinates: torch.Tensor, path: str) -> None:
        self.eval()
        output = self(coordinates)
        np.savetxt(path, output.cpu().detach().numpy())


class DisplacementFittingModel(FittingModel):
    # [x, y] -> [u_x, u_y]
    def __init__(self):
        super().__init__(
            in_num=COORDINATE_SIZE,
            out_num=DISPLACEMENT_SIZE,
            act_hidden=ACT_DISP,
            act_out=ACT_DISP_OUT,
        )

    def save_eval(
            self, coordinates: torch.Tensor,
            path_ux: str, path_uy: str,
    ):
        self.eval()
        displacement = self(coordinates)
        # assert isinstance(strain, torch.Tensor)
        u_x = displacement[:, 0]
        u_y = displacement[:, 1]

        makeDirIfNotExist(path_ux)
        makeDirIfNotExist(path_uy)
        np.savetxt(path_ux, u_x.cpu().detach().numpy())
        np.savetxt(path_uy, u_y.cpu().detach().numpy())


class StrainFittingModel(FittingModel):
    # [x, y] -> [ε_xx, ε_yy, γ_xy] (epsilon/e, epsilon/e, gamma/r)
    def __init__(self):
        super().__init__(
            in_num=COORDINATE_SIZE,
            out_num=STRAIN_SIZE,
            act_hidden=ACT_STRAIN,
            act_out=ACT_STRAIN_OUT,
        )

    def save_eval(
            self, coordinates: torch.Tensor,
            path_exx: str, path_eyy: str, path_rxy: str,
    ):
        self.eval()
        strain = self(coordinates)
        # assert isinstance(strain, torch.Tensor)
        e_xx = strain[:, 0]
        e_yy = strain[:, 1]
        r_xy = strain[:, 2]

        makeDirIfNotExist(path_exx)
        makeDirIfNotExist(path_eyy)
        makeDirIfNotExist(path_rxy)
        np.savetxt(path_exx, e_xx.cpu().detach().numpy())
        np.savetxt(path_eyy, e_yy.cpu().detach().numpy())
        np.savetxt(path_rxy, r_xy.cpu().detach().numpy())


# ---------------------------------------------------------------------------- #

# ============================= Elasticity Model ============================= #
class ElasticityModel(torch.nn.Module):
    # [x, y] -> [E, v]
    def __init__(self):
        super().__init__()

        self.num_layers = NUM_HIDDEN_LAYER_ELAS

        self.pos_encode = PositionalEncoding2D(D_POS_ENC_FIT)

        self.hidden1 = torch.nn.Linear(self.pos_encode.dim_out, NUM_NEURON_ELAS)
        for i in range(2, self.num_layers):
            setattr(self, f"hidden{i}", torch.nn.Linear(NUM_NEURON_ELAS, NUM_NEURON_ELAS))
        self.out = torch.nn.Linear(NUM_NEURON_ELAS, ELAS_SIZE)

        self.act1 = ACTIVATION_FUNCTION_ELAS()
        for i in range(2, self.num_layers):
            setattr(self, f"act{i}", ACTIVATION_FUNCTION_ELAS())
        self.act_out = ACTIVATION_FUNCTION_OUT_ELAS()
        self.act_out2 = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.pos_encode(x)
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")(getattr(self, f"hidden{i}")(x))
        if IS_NU_SIGMOID:
            x_new = x.clone()  # Clone x to preserve the original tensor for gradient computation
            x_new[:, 0] = self.act_out(x[:, 0])
            x_new[:, 1] = self.act_out2(x[:, 1]) / 2
            return x_new
        else:
            x = self.act_out(self.out(x))
            return x

    def init_weight_and_bias(layer):
        std_dev = 0.1
        max_dev = 2  # Maximum number of standard deviations from mean
        mean = 0
        initial_bias = 0.1
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.trunc_normal_(
                layer.weight,
                mean=mean,
                std=std_dev,
                a=(-max_dev * std_dev),
                b=(max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)

    def create_coordinate_tensor(self, coordinates: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            coordinates.reshape(-1, 2),
            dtype=TENSOR_TYPE,
            device=DEVICE,
            # requires_grad=True # TODO see if this is needed (later)
        )

    def save_eval(
            self, coordinates: torch.Tensor,
            path_E: str, path_v: str
    ):
        self.eval()
        elas = self(coordinates)
        # assert isinstance(strain, torch.Tensor)
        E = elas[:, 0]
        v = elas[:, 1]

        makeDirIfNotExist(path_E)
        makeDirIfNotExist(path_v)

        np.savetxt(path_E, E.cpu().detach().numpy())
        np.savetxt(path_v, v.cpu().detach().numpy())



class ElasticityEModel(torch.nn.Module):
    # [x, y] -> [E]
    def __init__(self):
        super().__init__()

        self.num_layers = NUM_HIDDEN_LAYER_ELAS

        self.pos_encode = PositionalEncoding2D(D_POS_ENC_FIT)

        self.hidden1 = torch.nn.Linear(self.pos_encode.dim_out, NUM_NEURON_ELAS)
        for i in range(2, self.num_layers):
            setattr(self, f"hidden{i}", torch.nn.Linear(NUM_NEURON_ELAS, NUM_NEURON_ELAS))
        self.out = torch.nn.Linear(NUM_NEURON_ELAS, ELAS_SIZE)

        self.act1 = ACTIVATION_FUNCTION_ELAS()
        for i in range(2, self.num_layers):
            setattr(self, f"act{i}", ACTIVATION_FUNCTION_ELAS())
        self.act_out = ACTIVATION_FUNCTION_OUT_ELAS()

    def forward(self, x: torch.Tensor):
        x = self.pos_encode(x)
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")(getattr(self, f"hidden{i}")(x))
        x = self.act_out(self.out(x))
        return x

    def init_weight_and_bias(layer):
        std_dev = 0.1
        max_dev = 2  # Maximum number of standard deviations from mean
        mean = 0
        initial_bias = 0.1
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.trunc_normal_(
                layer.weight,
                mean=mean,
                std=std_dev,
                a=(-max_dev * std_dev),
                b=(max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)

    def create_coordinate_tensor(self, coordinates: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            coordinates.reshape(-1, 2),
            dtype=TENSOR_TYPE,
            device=DEVICE,
            # requires_grad=True # TODO see if this is needed (later)
        )

    def save_eval(
            self, coordinates: torch.Tensor,
            path_E: str
    ):
        self.eval()
        elas = self(coordinates)
        # assert isinstance(strain, torch.Tensor)
        E = elas[:, 0]

        makeDirIfNotExist(path_E)

        np.savetxt(path_E, E.cpu().detach().numpy())


class ElasticityNuModel(torch.nn.Module):
    # [x, y] -> [E]
    def __init__(self):
        super().__init__()

        self.num_layers = NUM_HIDDEN_LAYER_ELAS

        self.pos_encode = PositionalEncoding2D(D_POS_ENC_FIT)

        self.hidden1 = torch.nn.Linear(self.pos_encode.dim_out, NUM_NEURON_ELAS)
        for i in range(2, self.num_layers):
            setattr(self, f"hidden{i}", torch.nn.Linear(NUM_NEURON_ELAS, NUM_NEURON_ELAS))
        self.out = torch.nn.Linear(NUM_NEURON_ELAS, ELAS_SIZE)

        self.act1 = ACTIVATION_FUNCTION_ELAS()
        for i in range(2, self.num_layers):
            setattr(self, f"act{i}", ACTIVATION_FUNCTION_ELAS())
        if IS_NU_SIGMOID:
            self.act_out = torch.nn.Sigmoid()
        else:
            self.act_out = ACTIVATION_FUNCTION_OUT_ELAS()

    def forward(self, x: torch.Tensor):
        x = self.pos_encode(x)
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")(getattr(self, f"hidden{i}")(x))
        x = self.act_out(self.out(x))/2
        return x

    def init_weight_and_bias(layer):
        std_dev = 0.1
        max_dev = 2  # Maximum number of standard deviations from mean
        mean = 0
        initial_bias = 0.1
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.trunc_normal_(
                layer.weight,
                mean=mean,
                std=std_dev,
                a=(-max_dev * std_dev),
                b=(max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)

    def create_coordinate_tensor(self, coordinates: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            coordinates.reshape(-1, 2),
            dtype=TENSOR_TYPE,
            device=DEVICE,
            # requires_grad=True # TODO see if this is needed (later)
        )

    def save_eval(
            self, coordinates: torch.Tensor,
            path_v: str
    ):
        self.eval()
        elas = self(coordinates)
        # assert isinstance(strain, torch.Tensor)
        v = elas[:, 0]

        makeDirIfNotExist(path_v)

        np.savetxt(path_v, v.cpu().detach().numpy())


# ============================= Parametric elasticity Model ============================= #
class IntensityFittingModel(FittingModel):
    # [x, y] -> [u_x, u_y]
    def __init__(self):
        super().__init__(
            in_num=COORDINATE_SIZE,
            out_num=INTENSITY_SIZE,
            act_hidden=ACT_INTENSITY,
            act_out=ACT_INTENSITY_OUT,
        )

    def save_eval(
            self, coordinates: torch.Tensor,
            path_I: str
    ):
        self.eval()
        intensity = self(coordinates)
        makeDirIfNotExist(path_I)
        np.savetxt(path_I, intensity.cpu().detach().numpy())


class ParametricElasticityModel(torch.nn.Module):
    # [x, y] -> [I]
    def __init__(self):
        super().__init__()

        if IS_COMPRESSIBLE:
            self.num_layers = NUM_HIDDEN_LAYER_ELAS

            self.pos_encode = PositionalEncoding2D(D_POS_ENC_FIT)

            self.hidden1 = torch.nn.Linear(self.pos_encode.dim_out, NUM_NEURON_ELAS)
            for i in range(2, self.num_layers):
                setattr(self, f"hidden{i}", torch.nn.Linear(NUM_NEURON_ELAS, NUM_NEURON_ELAS))
            self.out = torch.nn.Linear(NUM_NEURON_ELAS, ELAS_SIZE)

            self.act1 = ACTIVATION_FUNCTION_ELAS()
            for i in range(2, self.num_layers):
                setattr(self, f"act{i}", ACTIVATION_FUNCTION_ELAS())
            self.act_out = ACTIVATION_FUNCTION_OUT_ELAS()
        else:
            if FIXED_NU:
                self.v = torch.tensor(0.5, requires_grad=False)
            else:
                self.v = torch.nn.Parameter(torch.tensor(1.00))

        if PARAMETRIC_NEED_a:
            self.a = torch.nn.Parameter(torch.tensor(1.00))
        self.b = torch.nn.Parameter(torch.tensor(5.00))
        self.c = torch.nn.Parameter(torch.tensor(2.00))

    def forward(self, x: torch.Tensor):
        x = self.pos_encode(x)
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")(getattr(self, f"hidden{i}")(x))
        x = self.act_out(self.out(x))
        return x

    def init_weight_and_bias(layer):
        std_dev = 0.1
        max_dev = 2  # Maximum number of standard deviations from mean
        mean = 0
        initial_bias = 0.1
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.trunc_normal_(
                layer.weight,
                mean=mean,
                std=std_dev,
                a=(-max_dev * std_dev),
                b=(max_dev * std_dev))
            layer.bias.data.fill_(initial_bias)

    def create_coordinate_tensor(self, coordinates: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            coordinates.reshape(-1, 2),
            dtype=TENSOR_TYPE,
            device=DEVICE,
            # requires_grad=True # TODO see if this is needed (later)
        )

    if PARAMETRIC_NEED_a:
        def get_a(self):
            return torch.nn.functional.softplus(self.a)

    def get_b(self):
        return torch.nn.functional.softplus(self.b)

    def get_c(self):
        return torch.nn.functional.softplus(self.c)

    def set_nu(self, nu):
        self.v = torch.tensor(nu, requires_grad=False)

    def get_nu(self):
        if FIXED_NU:
            return self.v
        else:
            if TWODIM_LE_ASSUMPTION == 'planeStress' :
                return torch.nn.functional.sigmoid(self.v)
            elif TWODIM_LE_ASSUMPTION == 'planeStrain' :
                return torch.nn.functional.sigmoid(self.v)/0.5

    def save_eval(
            self, coordinates: torch.Tensor, intensity: torch.Tensor,
            path_E: str, path_v: str
    ):
        self.eval()
        if PARAMETRIC_NEED_a:
            E = self.get_a() * (( intensity + self.get_c())** self.get_b())
        else:
            E = (( intensity + self.get_c())** self.get_b())

        makeDirIfNotExist(path_E)
        np.savetxt(path_E, E.cpu().detach().numpy())

        if IS_COMPRESSIBLE:
            v = self(coordinates)
            # assert isinstance(strain, torch.Tensor)
            makeDirIfNotExist(path_v)
            np.savetxt(path_v, v.cpu().detach().numpy())

# ---------------------------------------------------------------------------- #

# ==================== Loss Components and Helper Modules ==================== #
class DataLoss(torch.nn.Module):
    # loss_d, equation (10) - modified.
    # Mean of the absolute difference (fitted - actual).
    def __init__(self):
        super().__init__()

    def forward(self, fitted_data: torch.Tensor, actual_data: torch.Tensor):
        if LOSS_NORM == "L1":
            return torch.mean(torch.abs(fitted_data - actual_data))
        else:
            return torch.mean((fitted_data - actual_data) ** 2)


class DiscrepancyLoss(DataLoss):
    def __init__(self):
        super().__init__()

    def forward(self, fitted_data: torch.Tensor, estimated_data: torch.Tensor):
        return super().forward(fitted_data=fitted_data, actual_data=estimated_data)


class ElasticityLoss(torch.nn.Module):
    # loss_E, equation (13).
    # Mean of the absolute difference with a constraint (fitted - constraint).
    def __init__(self):
        super().__init__()

    def forward(self, pred_E: torch.Tensor):
        if LOSS_NORM == "L1":
            return torch.abs(torch.mean(pred_E) - E_CONSTRAINT)
        else:
            return torch.mean((pred_E - E_CONSTRAINT)** 2)


class CalculateStrain(torch.nn.Module):
    # [u_x, u_y] -> [ε_xx, ε_yy, γ_xy]
    # Uses Finite Differentiation
    def __init__(self):
        super().__init__()

        self.conv_x = torch.nn.Conv2d(
            in_channels=1,  # only one data point at the pixel in the image
            out_channels=1,
            kernel_size=2,  # 2 by 2 square
            bias=False,
            stride=1,
            padding='valid')  # No +- value added to the kernel values
        self.conv_x.weight = torch.nn.Parameter(torch.tensor(
            [[-0.5, -0.5],
             [0.5, 0.5]],
            dtype=TENSOR_TYPE, device=DEVICE).reshape(1, 1, 2, 2))

        self.conv_y = torch.nn.Conv2d(
            in_channels=1,  # only one data point at the pixel in the image
            out_channels=1,
            kernel_size=2,  # 2 by 2 square
            bias=False,
            stride=1,
            padding='valid')  # No +- value added to the kernel values
        self.conv_y.weight = torch.nn.Parameter(torch.tensor(
            [[0.5, -0.5],
             [0.5, -0.5]],
            dtype=TENSOR_TYPE, device=DEVICE).reshape(1, 1, 2, 2))

    def forward(self, displacement: torch.Tensor, disp_coord: torch.Tensor, input_shape = STRAIN_INPUT_SHAPE):
        # Prepare the displacement values
        ux_mat = displacement[:, 0].reshape(1, 1, input_shape[0], input_shape[1])
        uy_mat = displacement[:, 1].reshape(1, 1, input_shape[0], input_shape[1])

        # Finite Differentiation using conv2d
        e_xx = self.conv_x(ux_mat)  # u_xx
        e_yy = self.conv_y(uy_mat)  # u_yy
        e_xy = self.conv_y(ux_mat) + self.conv_x(uy_mat)  # u_xy + u_yx

        # NOTE: From the paper the 100 constant
        e_xx = 100 * e_xx.reshape(-1)
        e_yy = 100 * e_yy.reshape(-1)
        e_xy = 100 * e_xy.reshape(-1)
        if SIMULATION_TYPE == "LOWRES" and not(FIT_STRAIN_HIGH_RES_CORD) and not (input_shape[0] == 257):
            e_xx = e_xx/DOWNSAMPLING_N
            e_yy = e_yy/DOWNSAMPLING_N
            e_xy = e_xy/DOWNSAMPLING_N

        return torch.stack([e_xx, e_yy, e_xy], dim=1)

    def evalStrain(self, displacement: torch.Tensor, disp_coord: torch.Tensor,input_shape = STRAIN_INPUT_SHAPE):
        # Prepare the displacement values
        ux_mat = displacement[:, 0].reshape(1, 1, input_shape[0], input_shape[1])
        uy_mat = displacement[:, 1].reshape(1, 1, input_shape[0], input_shape[1])

        # Finite Differentiation using conv2d
        e_xx = self.conv_x(ux_mat)  # u_xx
        e_yy = self.conv_y(uy_mat)  # u_yy
        e_xy = self.conv_y(ux_mat) + self.conv_x(uy_mat)  # u_xy + u_yx

        # NOTE: From the paper the 100 constant
        e_xx = 100 * e_xx.reshape(-1)
        e_yy = 100 * e_yy.reshape(-1)
        e_xy = 100 * e_xy.reshape(-1)
        if SIMULATION_TYPE == "LOWRES":
            e_xx = e_xx/DOWNSAMPLING_N
            e_yy = e_yy/DOWNSAMPLING_N
            e_xy = e_xy/DOWNSAMPLING_N

        return torch.stack([e_xx, e_yy, e_xy], dim=1)

class CalculateStrainAD(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, displacement: torch.Tensor, disp_coord: torch.Tensor):
        # Prepare the displacement values
        ux_tensor = displacement[:, 0].unsqueeze(dim=1)
        uy_tensor = displacement[:, 1].unsqueeze(dim=1)

        # Auto Differentiation
        Ux_deriv = torch.autograd.grad(
            ux_tensor, disp_coord,
            torch.ones([disp_coord.shape[0], 1], device=DEVICE),
            retain_graph=True,
            create_graph=True,
        )[0]

        Uy_deriv = torch.autograd.grad(
            uy_tensor, disp_coord,
            torch.ones([disp_coord.shape[0], 1], device=DEVICE),
            retain_graph=True,
            create_graph=True,
        )[0]


        e_xx = Ux_deriv[:, 0]  # u_xx
        e_yy = Uy_deriv[:, 1]  # u_yy
        e_xy = Ux_deriv[:, 1] + Uy_deriv[:, 0]  # u_xy + u_yx

        # NOTE: From the paper the 100 constant
        e_xx = 100 * e_xx.reshape(-1)
        e_yy = 100 * e_yy.reshape(-1)
        e_xy = 100 * e_xy.reshape(-1)

        return torch.stack([e_xx, e_yy, e_xy], dim=1)



class CalculateStress(torch.nn.Module):
    # [E, v, [ε_xx, ε_yy, γ_xy]] -> [σ_xx, σ_xy, 𝜏_xy]
    def __init__(self):
        super().__init__()

    def forward(self, E: torch.Tensor, v: torch.Tensor, strain: torch.Tensor):
        E_stack = torch.stack([E, E, E], dim=1)
        v_stack = torch.stack([v, v, v], dim=1)

        # Create a stack of c_matrices from the predicted v's
        c_stack = torch.stack([
            torch.ones(v.shape, dtype=TENSOR_TYPE, device=DEVICE),
            v,
            torch.zeros(v.shape, dtype=TENSOR_TYPE, device=DEVICE),  ##
            v,
            torch.ones(v.shape, dtype=TENSOR_TYPE, device=DEVICE),
            torch.zeros(v.shape, dtype=TENSOR_TYPE, device=DEVICE),  ##
            torch.zeros(v.shape, dtype=TENSOR_TYPE, device=DEVICE),
            torch.zeros(v.shape, dtype=TENSOR_TYPE, device=DEVICE),
            torch.divide(
                (torch.ones(v.shape, dtype=TENSOR_TYPE, device=DEVICE) - v),
                torch.full(v.shape, 2.0, dtype=TENSOR_TYPE, device=DEVICE)
            )  ##
        ], dim=1).reshape([-1, 3, 3])

        # Squeeze as output is a stack of 1 by 3 matrices, but want stack
        # of dim=3 vectors. The squeeze is remove dimensions of size
        # 1 resulting from the strict matrix multiplication.(1x3)(3x3)
        # C(v(x,y)) * Strain(x,y)
        matmul_results = torch.bmm(strain.reshape(-1, 1, 3), c_stack).squeeze()

        # The fraction out front
        # TODO remove E
        fraction = torch.divide(E_stack, 1 - torch.square(v_stack))

        # Fraction * (C*strain)
        stress = torch.multiply(matmul_results, fraction)
        return stress


# The portion of loss that uses Auto-Differentiation
class EquilibriumLoss(torch.nn.Module):
    # loss_r, based on equation (8) and the equations that composes its parts.
    def __init__(self):
        super().__init__()

        # For pred_E sum, equation (9). self.calculate_E_hat
        self.sum_kernel = torch.tensor(
            [[1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0], ],
            dtype=TENSOR_TYPE, device=DEVICE
        )

        # For the equilibrium condition finite differentiation
        self.w_conv_x = torch.tensor(
            [[-1.0, -1.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0], ],
            dtype=TENSOR_TYPE, device=DEVICE
        )
        self.w_conv_y = torch.tensor(
            [[1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0], ],
            dtype=TENSOR_TYPE, device=DEVICE
        )

    # Must pass the coordinates used to generate the predicted values
    def forward(self, pred_E: torch.Tensor, stress: torch.Tensor, elas_shape = ELAS_INPUT_SHAPE ):
        # Prepare the stress values for self.conv2d
        sxx_mat = stress[:, 0].reshape(1, 1, elas_shape[0], elas_shape[1])
        syy_mat = stress[:, 1].reshape(1, 1, elas_shape[0], elas_shape[1])
        sxy_mat = stress[:, 2].reshape(1, 1, elas_shape[0], elas_shape[1])

        # Finite Differentiation using conv2d
        sxx_x = self.conv2d(sxx_mat, self.w_conv_x)
        syy_y = self.conv2d(syy_mat, self.w_conv_y)
        sxy_x = self.conv2d(sxy_mat, self.w_conv_x)
        sxy_y = self.conv2d(sxy_mat, self.w_conv_y)

        # Equilibrium Condition, equation (6).
        f_x = sxx_x + sxy_y
        f_y = syy_y + sxy_x

        # Normalize the losses, equation (8).
        E_hat_pred = self.calculate_E_hat(pred_E,elas_shape)
        f_x_norm = f_x / E_hat_pred
        f_y_norm = f_y / E_hat_pred

        if LOSS_NORM == "L1":
            # Loss of in each coordinate: L1
            loss_x = torch.mean(torch.abs(f_x_norm))
            loss_y = torch.mean(torch.abs(f_y_norm))
        else:
            # Loss of in each coordinate: L2
            loss_x = torch.mean((f_x_norm)** 2)/1000
            loss_y = torch.mean((f_y_norm)** 2)/1000

        return loss_x + loss_y

    # Used for Finite Differentiation (requires specific 4D tensor)
    def conv2d(self, x, W: torch.Tensor):
        W = W.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        return torch.nn.functional.conv2d(
            x, W,
            stride=1,
            padding='valid'
        )

    # E_hat_pred equation (9)
    def calculate_E_hat(self, pred_E: torch.Tensor,elas_shape=ELAS_INPUT_SHAPE ) -> torch.Tensor:
        pred_E_matrix = torch.reshape(pred_E, [elas_shape[0], elas_shape[1]])
        pred_E_matrix_4d = torch.reshape(pred_E_matrix, [-1, 1, elas_shape[0], elas_shape[1]])
        pred_E_conv = self.conv2d(pred_E_matrix_4d, self.sum_kernel)
        return pred_E_conv


class EquilibriumLossAD(torch.nn.Module):
    # loss_r, based on equation (8) and the equations that composes its parts.
    def __init__(self):
        super().__init__()

    # Must pass the coordinates used to generate the predicted values
    def forward(self, pred_E: torch.Tensor, stress: torch.Tensor, strain_coord: torch.Tensor):
        s_xx = stress[:, 0].unsqueeze(dim=1)
        s_yy = stress[:, 1].unsqueeze(dim=1)
        s_xy = stress[:, 2].unsqueeze(dim=1)

        # TODO: 2) do d/dx (log(E(x)))
        # Will also change stress calculations

        # Calculate Gradients using auto-differentiation
        f_xx_x_y = torch.autograd.grad(
            s_xx, strain_coord,
            torch.ones([strain_coord.shape[0], 1], device=DEVICE),
            retain_graph=True,
            # create_graph=True,
        )[0]
        f_xy_x_y = torch.autograd.grad(
            s_xy, strain_coord,
            torch.ones([strain_coord.shape[0], 1], device=DEVICE),
            retain_graph=True,
            # create_graph=True,
        )[0]
        f_yy_x_y = torch.autograd.grad(
            s_yy, strain_coord,
            torch.ones([strain_coord.shape[0], 1], device=DEVICE),
            retain_graph=True,
            # create_graph=True,
        )[0]

        # Equilibrium Condition, equation (6).
        # TODO: move [:, 0] to its own function for readability
        f_x = f_xx_x_y[:, 0] + f_xy_x_y[:, 1]
        f_y = f_yy_x_y[:, 1] + f_xy_x_y[:, 0]

        # Normalize the losses, equation (8).>> Just added
        E_hat_pred = torch.mean(pred_E)
        f_x_norm = f_x / E_hat_pred
        f_y_norm = f_y / E_hat_pred

        if LOSS_NORM == "L1":
            # Loss of in each coordinate: L1
            loss_x = torch.mean(torch.abs(f_x_norm))
            loss_y = torch.mean(torch.abs(f_y_norm))
        else:
            # Loss of in each coordinate: L2
            loss_x = torch.mean((f_x_norm)** 2)/1000
            loss_y = torch.mean((f_y_norm)** 2)/1000

        return loss_x + loss_y


# ---------------------------------------------------------------------------- #

# ======================= Inverse Problem "Controller" ======================= #
class InverseProblem():
    # Trained in 3 stages.
    # Fit 1:
    # - Displacement Fit, u_nn, from displacement data.
    # Fit 2:
    # - Strain Fit, s_nn, using s_u.
    # - s_u: FD of u_nn -> s_u (calculate strain)
    # Train Elasticity:
    # - Elasticity Model, E_nn.
    # - FD: s_nn -> stress -> residuals (equilibrium condition)

    def __init__(
            self,
            disp_coordinates: np.ndarray,
            disp_coord_array_for_eval: np.ndarray,
            strain_coordinates: np.ndarray,
            disp_data: np.ndarray,
            elas_coordinates: np.ndarray,
            elas_coord_array_for_eval: np.ndarray,
            data_e: np.ndarray,
            data_nu: np.ndarray,
            neptune=None,
    ):
        # CALCULATE FORCE EXTERNAL
        if SCALING_E:
            self.force_data = CalculateForceFromDAT(DATAFOLDER, DATASETNAME)
            print(f'The calculated external force applied to boundary is {self.force_data:.4f} N')

        # Initialize the component PyTorch Modules.
        # Models
        self.disp_fit_model = DisplacementFittingModel()
        self.disp_fit_model.to(DEVICE)
        self.disp_fit_model.apply(DisplacementFittingModel.init_weight_and_bias)

        self.strain_fit_model = StrainFittingModel()
        self.strain_fit_model.to(DEVICE)
        self.strain_fit_model.apply(StrainFittingModel.init_weight_and_bias)

        if SEPARATED_ENu:
            if PARAMETRIC_E:
                self.intensity_fit_model = IntensityFittingModel()
                self.intensity_fit_model.to(DEVICE)
                self.intensity_fit_model.apply(IntensityFittingModel.init_weight_and_bias)
                self.elas_model = ParametricElasticityModel()
                self.elas_model.to(DEVICE)
                self.elas_model.apply(ParametricElasticityModel.init_weight_and_bias)
            else:
                self.elas_model = ElasticityEModel()
                self.elas_model.to(DEVICE)
                self.elas_model.apply(ElasticityEModel.init_weight_and_bias)
                self.nu_model = ElasticityNuModel()
                self.nu_model.to(DEVICE)
                self.nu_model.apply(ElasticityNuModel.init_weight_and_bias)
        else:
            # This is non-parametric E
            self.elas_model = ElasticityModel()
            self.elas_model.to(DEVICE)
            self.elas_model.apply(ElasticityModel.init_weight_and_bias)

        # For saving sake
        self.fit_1_epochs = 0

        # Various Components of Loss
        self.data_loss = DataLoss()
        self.discrepancy_loss = DiscrepancyLoss()
        self.elas_loss = ElasticityLoss()
        if DIFFERTIATION_METHOD == "FD":
            self.calc_strain = CalculateStrain()
        else:
            self.calc_strain = CalculateStrainAD()
        self.calc_stress = CalculateStress()
        if DIFFERTIATION_METHOD == "FD":
            self.eq_loss = EquilibriumLoss()
        else:
            self.eq_loss = EquilibriumLossAD()

        # Optimizer depends on what kind of training
        self.optimizer_class = torch.optim.Adam

        # Create Coordinates. Copy in case the tensors would refer to the same memory.
        self.disp_coord = self.disp_fit_model.create_coordinate_tensor(disp_coordinates.copy())
        self.strain_coord = self.strain_fit_model.create_coordinate_tensor(strain_coordinates.copy())
        self.elas_coord = self.elas_model.create_coordinate_tensor(elas_coordinates.copy())
        self.elas_coord_for_eval = self.elas_model.create_coordinate_tensor(elas_coord_array_for_eval.copy())
        self.disp_coord_for_eval = self.elas_model.create_coordinate_tensor(disp_coord_array_for_eval.copy())

        # Fitting Data
        self.disp_data = torch.tensor(disp_data, dtype=TENSOR_TYPE, device=DEVICE)
        self.data_e = data_e
        self.data_nu = data_nu

        if not DEVELOPMENT_MODE:
            self.neptune = neptune

        #Set parametric E
        if PARAMETRIC_E:
            m_data_array = np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/m_data')
            ct_data_array = np.exp((1 / E_SET_b) * np.log(m_data_array / E_SET_a)) - E_SET_c
            # Remarks
            # If input is E >> No need to rescale for inverse E = I However, make sure that I is in the interval 0 to 1
            # But if the input is Intensity >> it is required to rescale it
            # The current data I is between 0 to 1 so, it don't need to normalize
            int_data_array = ct_data_array
            self.int_data = torch.tensor(int_data_array, dtype=TENSOR_TYPE, device=DEVICE).unsqueeze(1)

            if FIXED_NU and not IS_COMPRESSIBLE:
                nu_data_array = np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/nu_data')
                self.elas_model.set_nu(nu_data_array[0])


    # ---------------------------- Load Saved Models ----------------------------- #
    # Returns True if model was found and loaded.
    # Path is within the model subfolder
    def load_pretrained_displacement(self, file_path: str = f"{PRE_FIT_MODEL_SUBFOLDER}/disp") -> bool:
        try:
            self.disp_fit_model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt", weights_only=True))
            return True
        except:
            return False

    def load_pretrained_model_displacement(self, file_path: str = f"{PRE_FIT_MODEL_SUBFOLDER}/disp") -> bool:
        try:
            self.disp_fit_model.load_state_dict(torch.load(f"{OUTPUT_FOLDER_MODEL}{MODEL_SUBFOLDER}/{file_path}.pt", weights_only=True))
            return True
        except:
            return False

    # Returns True if model was found and loaded.
    # Path is within the model subfolder
    def load_pretrained_strain(self, file_path: str) -> bool:
        try:
            self.strain_fit_model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt", weights_only=True))
            return True
        except:
            return False

    def load_pretrained_model_strain(self, file_path: str) -> bool:
        try:
            self.strain_fit_model.load_state_dict(torch.load(f"{OUTPUT_FOLDER_MODEL}{MODEL_SUBFOLDER}/{file_path}.pt", weights_only=True))
            return True
        except:
            return False

    # Returns True if model was found and loaded.
    # Path is within the model subfolder
    def load_pretrained_intensity(self, file_path: str) -> bool:
        try:
            self.intensity_fit_model.load_state_dict(
                torch.load(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt", weights_only=True))
            return True
        except:
            return False
    def load_pretrained_model_intensity(self, file_path: str) -> bool:
        try:
            self.intensity_fit_model.load_state_dict(
                torch.load(f"{OUTPUT_FOLDER_MODEL}{MODEL_SUBFOLDER}/{file_path}.pt", weights_only=True))
            return True
        except:
            return False

    # Returns True if model was found and loaded.
    # Path is within the model subfolder
    def load_pretrained_elas(self, file_path: str) -> bool:
        try:
            self.elas_model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt", weights_only=True))
            return True
        except:
            return False


    # --------------------------- Save Trained Models ---------------------------- #
    def save_displacement_model(self, file_path: str = f"disp") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.disp_fit_model.state_dict(), f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")

    def save_displacement_pretrained_model(self, file_path: str = f"disp") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER_MODEL}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.disp_fit_model.state_dict(), f"{OUTPUT_FOLDER_MODEL}{MODEL_SUBFOLDER}/{file_path}.pt")

    def save_strain_model(self, file_path: str = "strain") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.strain_fit_model.state_dict(), f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")

    def save_strain_pretrained_model(self, file_path: str = "strain") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER_MODEL}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.strain_fit_model.state_dict(), f"{OUTPUT_FOLDER_MODEL}{MODEL_SUBFOLDER}/{file_path}.pt")

    def save_intensity_model(self, file_path: str = "intensity") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.intensity_fit_model.state_dict(), f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")

    def save_intensity_pretrained_model(self, file_path: str = "intensity") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER_MODEL}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.intensity_fit_model.state_dict(), f"{OUTPUT_FOLDER_MODEL}{MODEL_SUBFOLDER}/{file_path}.pt")

    def save_elas_model(self, file_path: str = "elas") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.elas_model.state_dict(), f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")

    def save_nu_model(self, file_path: str = "nu") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.nu_model.state_dict(), f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")

    # ---------------------- Error function for Each stage ----------------------- #
    def fit_1_error(self, disp_fit: torch.Tensor) -> torch.Tensor:
        return torch.mean((disp_fit - self.disp_data) ** 2)

    def fit_3_error(self, int_fit: torch.Tensor) -> torch.Tensor:
        return torch.mean((int_fit - self.int_data) ** 2)

    # ---------------------- Loss Functions for Each Stage ----------------------- #
    def fit_1_loss(self, disp_fit: torch.Tensor ) -> torch.Tensor:
        data_loss = self.data_loss(
                fitted_data=disp_fit,
                actual_data=self.disp_data
            )
        total_data_loss = data_loss * WEIGHT_D_DISP
        return total_data_loss

    def fit_2_loss(
            self,
            disp_fit: torch.Tensor,
            strain_fit: torch.Tensor,
            loss_list: list = None
    ) -> torch.Tensor:
        if DIFFERTIATION_METHOD == "AD":
            strain_coord_grad = self.elas_coord_for_eval.requires_grad_()
            disp_fit_strain = self.disp_fit_model(strain_coord_grad)
            loss_d_strain = self.discrepancy_loss(
                fitted_data=strain_fit,
                estimated_data=self.calc_strain(displacement=disp_fit_strain,
                                                disp_coord=strain_coord_grad)
            )
        else:
            loss_d_strain = self.discrepancy_loss(
                fitted_data=strain_fit,
                estimated_data=self.calc_strain(displacement=disp_fit,
                                                disp_coord=self.disp_coord)
            )
        loss_d_disp = self.data_loss(
            fitted_data=disp_fit,
            actual_data=self.disp_data
        )

        loss = loss_d_strain * WEIGHT_D_STRAIN + loss_d_disp * WEIGHT_D_DISP
        if loss_list != None:
            loss_list += [
                (loss_d_strain * WEIGHT_D_STRAIN).item(),
                (loss_d_disp * WEIGHT_D_DISP).item(),
                loss.item()
            ]

        return loss

    def fit_2_loss_highRES(
            self,
            disp_fit: torch.Tensor,
            disp_pred: torch.Tensor,
            strain_fit: torch.Tensor,
            loss_list: list = None
    ) -> torch.Tensor:

        loss_d_strain = self.discrepancy_loss(
            fitted_data=strain_fit,
            estimated_data=self.calc_strain(displacement=disp_pred,
                                            disp_coord=self.disp_coord_for_eval)
        )
        loss_d_disp = self.data_loss(
            fitted_data=disp_fit,
            actual_data=self.disp_data
        )

        loss = loss_d_strain * WEIGHT_D_STRAIN + loss_d_disp * WEIGHT_D_DISP
        if loss_list != None:
            loss_list += [
                (loss_d_strain * WEIGHT_D_STRAIN).item(),
                (loss_d_disp * WEIGHT_D_DISP).item(),
                loss.item()
            ]

        self.neptune.append_value_to_field("fit2/iteration/loss_disp", loss_d_disp)
        self.neptune.append_value_to_field("fit2/iteration/loss_strain", loss_d_strain)
        self.neptune.append_value_to_field("fit2/iteration/loss", loss)

        return loss


    def fit_3_loss(self, int_fit: torch.Tensor) -> torch.Tensor:
        return self.data_loss(
                fitted_data=int_fit,
                actual_data=self.int_data
            ) * WEIGHT_INT

    # loss_list is for the current iteration. Will append to whatever is provided
    # before ~ [ num_epoch ]
    # after = [ e, wdd*ldd. wds*lds, we*le, wr*lr, l_total]
    def train_elas_loss(
            self,
            disp_fit: torch.Tensor,
            strain_fit: torch.Tensor,
            elas_pred: torch.Tensor,
            strain_fit_elas_coord: torch.Tensor,
            loss_list: list = None
    ) -> torch.Tensor:

        if FIT_DISPLACEMENT:
            if FIT_STRAIN:
                if DIFFERTIATION_METHOD == "AD":
                    strain_coord_grad = self.elas_coord_for_eval.requires_grad_()
                    disp_fit_strain = self.disp_fit_model(strain_coord_grad)
                    loss_d_strain = self.discrepancy_loss(
                        fitted_data=strain_fit,
                        estimated_data=self.calc_strain(displacement=disp_fit_strain,
                                                        disp_coord=strain_coord_grad)
                    )
                else:
                    if FIT_STRAIN_HIGH_RES_CORD:
                        disp_fit_high = self.disp_fit_model(self.disp_coord_for_eval)
                        strain_fit_for_eval = self.strain_fit_model(self.elas_coord_for_eval)
                        loss_d_strain = self.discrepancy_loss(
                            fitted_data=strain_fit_for_eval ,
                            estimated_data=self.calc_strain(displacement=disp_fit_high,
                                                            disp_coord=self.disp_coord_for_eval)
                        )
                    else:
                        loss_d_strain = self.discrepancy_loss(
                            fitted_data=strain_fit,
                            estimated_data=self.calc_strain(displacement=disp_fit,
                                                            disp_coord=self.disp_coord)
                        )
            loss_d_disp = self.data_loss(
                fitted_data=disp_fit,
                actual_data=self.disp_data
            )
        else:
            if FIT_STRAIN:
                loss = self.discrepancy_loss(
                    fitted_data=strain_fit,
                    estimated_data=self.strain_data
                )
            else:
                strain_fit_elas_coord = self.calc_strain(displacement= self.disp_data,
                                                            disp_coord=self.disp_coord)

        # Unpack Elasticities
        E_pred = elas_pred[:, 0]
        v_pred = elas_pred[:, 1]

        # Elasticity Loss
        loss_E = self.elas_loss(E_pred)

        # Calculate Stress
        stress = self.calc_stress(
            E=E_pred,
            v=v_pred,
            strain=strain_fit_elas_coord,
        )

        # Equilibrium Loss
        if DIFFERTIATION_METHOD == "FD":
            if not(EVALUATE_ELAS_HIGH_RES) and SIMULATION_TYPE == "LOWRES":
                loss_r = self.eq_loss(
                    pred_E=E_pred,
                    stress=stress,
                    elas_shape = torch.tensor([128, 128], device=DEVICE)
                )
            else:
                loss_r = self.eq_loss(
                    pred_E=E_pred,
                    stress=stress
                )
        else:
            loss_r = self.eq_loss(
                pred_E=E_pred,
                stress=stress,
                strain_coord=self.elas_coord.requires_grad_()
            )

        if FIT_DISPLACEMENT:
            if FIT_STRAIN:
                loss = loss_d_strain * WEIGHT_D_STRAIN + loss_d_disp * WEIGHT_D_DISP
            else:
                loss = loss_d_disp * WEIGHT_D_DISP
        elif FIT_STRAIN:
            loss = loss_d_strain * WEIGHT_D_STRAIN
        if LOSS_MEAN_E:
            loss += loss_E * WEIGHT_E + loss_r * WEIGHT_R
        else:
            loss += loss_r * WEIGHT_R

        if INVESTIGATION_MODE and FIT_DISPLACEMENT and FIT_STRAIN:
            if math.isnan(loss_d_disp.item()) or math.isnan(loss_d_strain.item()) or math.isnan(loss_E.item()) or math.isnan(loss_r.item()) :  # Check if the value is NaN
                print(f"NaN detected")
                print("Execution paused. Press Enter to continue.")
                input()

        if loss_list != None:
            if FIT_DISPLACEMENT:
                if FIT_STRAIN:
                    current_loss = [
                        (loss_d_disp * WEIGHT_D_DISP).item(),
                        (loss_d_strain * WEIGHT_D_STRAIN).item(),
                        (loss_E * WEIGHT_E).item(),
                        (loss_r * WEIGHT_R).item(),
                        loss.item()
                    ]
                else:
                    current_loss = [
                        (loss_d_disp * WEIGHT_D_DISP).item(),
                        (loss_E * WEIGHT_E).item(),
                        (loss_r * WEIGHT_R).item(),
                        loss.item()
                    ]

            else:
                current_loss = [
                    (loss_E * WEIGHT_E).item(),
                    (loss_r * WEIGHT_R).item(),
                    loss.item()
                ]
            loss_list += current_loss

            # Check if previous_loss is defined; initialize if not
            if 'previous_loss' not in locals():
                previous_loss = current_loss

            # Initialize a list to track NaN losses
            nan_losses = []

            flag = 0
            # Check for jumps and NaNs
            for i, (current, previous) in enumerate(zip(current_loss, previous_loss)):
                if math.isnan(current):  # Check for NaN
                    nan_losses.append(NAMES_LOSS[i])  # Record loss name with NaN
                elif previous > 0 and current > 5 * previous:  # Check for jump
                    warnings.warn(
                        f"{NAMES_LOSS[i]} jumped from {previous:.6f} to {current:.6f}, more than 5 times."
                    )
                    flag = 1

            # After iteration, check if any NaNs were detected
            if nan_losses:
                warnings.warn(f"The following losses are NaN: {', '.join(nan_losses)}")
                self.neptune.add_group_tags("ERROR:NAN")
                self.neptune.add_tags("ERROR:NAN")
                # RUN["sys/group_tags"].add("ERROR:NAN")
                # RUN["sys/tags"].add("ERROR:NAN")
                print("Detected NaN values in the loss computation.")
                warnings.warn("Detected NaN values in the loss computation.")
                raise ValueError("Detected NaN values in the loss computation.")
                flag = 2

            previous_loss = current_loss

        if FIT_DISPLACEMENT:
            self.neptune.append_value_to_field("train/iteration/loss_d_disp", loss_d_disp)
            if FIT_STRAIN:
                self.neptune.append_value_to_field("train/iteration/loss_d_strain", loss_d_strain)
        self.neptune.append_value_to_field("train/iteration/loss_E", loss_E)
        self.neptune.append_value_to_field("train/iteration/loss_r", loss_r)
        self.neptune.append_value_to_field("train/iteration/loss", loss)
        # RUN["train/iteration/loss_d_disp"].append(loss_d_disp)
        # RUN["train/iteration/loss_d_strain"].append(loss_d_strain)
        # RUN["train/iteration/loss_E"].append(loss_E)
        # RUN["train/iteration/loss_r"].append(loss_r)
        # RUN["train/iteration/loss"].append(loss)

        return loss
    def train_parametric_elas_loss(
            self,
            disp_fit: torch.Tensor,
            strain_fit: torch.Tensor,
            E_pred: torch.Tensor,
            v_pred: torch.Tensor,
            strain_fit_elas_coord: torch.Tensor,
            loss_list: list = None,
    ) -> torch.Tensor:

        if FIT_DISPLACEMENT:
            if DIFFERTIATION_METHOD == "AD":
                strain_coord_grad = self.strain_coord.requires_grad_()
                disp_fit_strain = self.disp_fit_model(strain_coord_grad)
                loss_d_strain = self.discrepancy_loss(
                    fitted_data=strain_fit,
                    estimated_data=self.calc_strain(displacement=disp_fit_strain,
                                                    disp_coord=strain_coord_grad)
                )
            else:
                loss_d_strain = self.discrepancy_loss(
                    fitted_data=strain_fit,
                    estimated_data=self.calc_strain(displacement=disp_fit,
                                                disp_coord=self.disp_coord)
                )
            loss_d_disp = self.data_loss(
                fitted_data=disp_fit,
                actual_data=self.disp_data
            )
        else:
            strain_fit_elas_coord = self.calc_strain(displacement= self.disp_data,
                                            disp_coord=self.disp_coord)

        # Intensity loss
        int_fit = self.intensity_fit_model(self.strain_coord)
        loss_int = self.data_loss(
                fitted_data=int_fit,
                actual_data=self.int_data)

        # Elasticity Loss
        loss_E = self.elas_loss(E_pred)

        # Calculate Stress
        stress = self.calc_stress(
            E=E_pred,
            v=v_pred,
            strain=strain_fit_elas_coord,
        )

        # Equilibrium Loss
        if DIFFERTIATION_METHOD == "FD":
            loss_r = self.eq_loss(
                pred_E=E_pred,
                stress=stress
            )
        else:
            loss_r = self.eq_loss(
                pred_E=E_pred,
                stress=stress,
                strain_coord=self.elas_coord.requires_grad_()
            )


        if FIT_DISPLACEMENT:
            loss = loss_d_strain * WEIGHT_D_STRAIN + loss_d_disp * WEIGHT_D_DISP + loss_int * WEIGHT_INT
            if LOSS_MEAN_E:
                loss += loss_E * WEIGHT_E + loss_r * WEIGHT_R
            else:
                loss +=  loss_r * WEIGHT_R
        else:
            if LOSS_MEAN_E:
                loss = loss_E * WEIGHT_E + loss_r * WEIGHT_R + loss_int * WEIGHT_INT
            else:
                loss = loss_r * WEIGHT_R + loss_int * WEIGHT_INT

        if loss_list != None:
            if FIT_DISPLACEMENT:
                current_loss = [
                    (loss_d_disp * WEIGHT_D_DISP).item(),
                    (loss_d_strain * WEIGHT_D_STRAIN).item(),
                    (loss_E * WEIGHT_E).item(),
                    (loss_r * WEIGHT_R).item(),
                    (loss_int * WEIGHT_INT).item(),
                    loss.item()
                ]
            else:
                current_loss = [
                    (loss_E * WEIGHT_E).item(),
                    (loss_r * WEIGHT_R).item(),
                    (loss_int * WEIGHT_INT).item(),
                    loss.item()
                ]
            loss_list += current_loss

            # Check if previous_loss is defined; initialize if not
            if 'previous_loss' not in locals():
                previous_loss = current_loss

            # Initialize a list to track NaN losses
            nan_losses = []

            flag = 0
            # Check for jumps and NaNs
            for i, (current, previous) in enumerate(zip(current_loss, previous_loss)):
                if math.isnan(current):  # Check for NaN
                    nan_losses.append(NAMES_LOSS[i])  # Record loss name with NaN
                elif previous > 0 and current > 5 * previous:  # Check for jump
                    warnings.warn(
                        f"{NAMES_LOSS[i]} jumped from {previous:.6f} to {current:.6f}, more than 5 times."
                    )
                    flag = 1

            # After iteration, check if any NaNs were detected
            if nan_losses:
                warnings.warn(f"The following losses are NaN: {', '.join(nan_losses)}")
                self.neptune.add_group_tags("ERROR:NAN")
                self.neptune.add_tags("ERROR:NAN")
                # RUN["sys/group_tags"].add("ERROR:NAN")
                # RUN["sys/tags"].add("ERROR:NAN")
                print("Detected NaN values in the loss computation.")
                warnings.warn("Detected NaN values in the loss computation.")
                raise ValueError("Detected NaN values in the loss computation.")
                flag = 2

            previous_loss = current_loss

        if FIT_DISPLACEMENT:
            self.neptune.append_value_to_field("train/iteration/loss_d_disp", loss_d_disp)
            self.neptune.append_value_to_field("train/iteration/loss_d_strain", loss_d_strain)
        self.neptune.append_value_to_field("train/iteration/loss_I", loss_int)
        self.neptune.append_value_to_field("train/iteration/loss_E", loss_E)
        self.neptune.append_value_to_field("train/iteration/loss_r", loss_r)
        self.neptune.append_value_to_field("train/iteration/loss", loss)

        self.neptune.append_value_to_field("train/iteration/b", (self.elas_model.get_b()).item())
        self.neptune.append_value_to_field("train/iteration/c", (self.elas_model.get_c()).item())

        # RUN["train/iteration/loss_d_disp"].append(loss_d_disp)
        # RUN["train/iteration/loss_d_strain"].append(loss_d_strain)
        # RUN["train/iteration/loss_E"].append(loss_E)
        # RUN["train/iteration/loss_r"].append(loss_r)
        # RUN["train/iteration/loss"].append(loss)

        return loss

    # ---------------------- Train Functions for Each Stage ---------------------- #
    def run_fit_1(self, num_epochs=NUM_FITTING_1_EPOCHS) -> None:
        optimizer = self.optimizer_class(self.disp_fit_model.parameters(), lr=LEARN_RATE)

        if STATE_MESSAGES: print("STATE: Starting Fit 1 - Displacement Fitting.")
        training_start_time = time.time()

        for e in range(num_epochs):
            print(f"Fit-1 Epoch {e} Starting.")
            epoch_start_time = time.time()

            self.disp_fit_model.train()
            for i in range(1, ITERATIONS_PER_EPOCH + 1):
                optimizer.zero_grad()
                disp_fit = self.disp_fit_model(self.disp_coord)
                loss_fit = self.fit_1_loss(disp_fit=disp_fit)
                error_fit = self.fit_1_error(disp_fit=disp_fit)
                loss = loss_fit

                loss.backward()
                optimizer.step()

                if i % NOTIFY_ITERATION_MOD == 0:
                    print(
                        f"Fit-1 Epoch: {e} [{i}/{ITERATIONS_PER_EPOCH} ({1.0 * i / ITERATIONS_PER_EPOCH * 100:.2f}%)]\tLoss: {loss.item():.6f}")

            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"Epoch{e} took {epoch_elapsed_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(num_epochs - e) * epoch_elapsed_time)}")
            self.neptune.append_value_to_field("fit1/epoch/loss", loss)
            self.neptune.append_value_to_field("fit1/epoch/loss_fit", loss_fit)
            self.neptune.append_value_to_field("fit1/epoch/error_fit1", error_fit)
            # RUN["fit1/epoch/loss"].append(loss)

            self.disp_fit_model.save_eval(
                self.disp_coord,
                path_ux=f"{OUTPUT_FOLDER}/pred_ux/fit1-{e}.txt",
                path_uy=f"{OUTPUT_FOLDER}/pred_uy/fit1-{e}.txt",
            )

            self.fit_1_epochs += 1

        # Visualize U pred
        u_pred = self.disp_fit_model(self.disp_coord_for_eval)
        data_disp = np.loadtxt(f"../data/{DATAFOLDER}/{DATASETNAME}/disp_data")
        if IS_DIM_PREDEFINED:
            dim = int(np.sqrt(data_disp.shape[0]))
            data_ux = data_disp[:, 0].reshape(dim, dim)
            data_uy = data_disp[:, 1].reshape(dim, dim)

            dim = int(np.sqrt(u_pred[:, 0].shape[0]))
            ux_data = u_pred[:, 0].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
            uy_data = u_pred[:, 1].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
        else:
            data_ux = data_disp[:, 0].reshape(NUM_ROWS_ALL, NUM_COLS_ALL)
            data_uy = data_disp[:, 1].reshape(NUM_ROWS_ALL, NUM_COLS_ALL)
            ux_data = u_pred[:, 0].detach().cpu().numpy().reshape(NUM_ROWS_ALL,
                                                                  NUM_COLS_ALL)  # Transfer to CPU and convert to NumPy
            uy_data = u_pred[:, 1].detach().cpu().numpy().reshape(NUM_ROWS_ALL,
                                                                  NUM_COLS_ALL)  # Transfer to CPU and convert to NumPy

        if (SIMULATION_TYPE == "LOWRES"):
            ux_data = np.flip(ux_data, axis=1)
            uy_data = np.flip(uy_data, axis=1)
        neptune_plot2d(data_ux, "true/plot/ux_true", neptune=self.neptune)
        neptune_plot2d(data_uy, "true/plot/uy_true", neptune=self.neptune)

        neptune_plot2d(ux_data, f"fit1/plot/ux_pred", neptune=self.neptune)
        neptune_plot2d(uy_data, f"fit1/plot/uy_pred", neptune=self.neptune)

        neptune_plot2d((data_ux - ux_data), f"fit1/plot/ux_error", neptune=self.neptune)
        neptune_plot2d((data_uy - uy_data), f"fit1/plot/uy_error", neptune=self.neptune)

        VISUALIZE_E_DERIVU = True
        if VISUALIZE_E_DERIVU:
            disp_pred = self.disp_fit_model(self.disp_coord_for_eval)
            e_from_u = self.calc_strain(displacement=disp_pred,
                                        disp_coord=self.disp_coord_for_eval,input_shape= torch.tensor([257, 257], device=DEVICE))
            if IS_DIM_PREDEFINED:
                dim = int(np.sqrt(self.elas_coord_for_eval[:, 0].shape[0]))
                exx_from_u = e_from_u[:, 0].detach().cpu().numpy().reshape(dim,
                                                                           dim)  # Transfer to CPU and convert to NumPy
                eyy_from_u = e_from_u[:, 1].detach().cpu().numpy().reshape(dim,
                                                                           dim)  # Transfer to CPU and convert to NumPy
                rxy_from_u = e_from_u[:, 2].detach().cpu().numpy().reshape(dim,
                                                                           dim)  # Transfer to CPU and convert to NumPy
            else:
                exx_from_u = e_from_u[:, 0].detach().cpu().numpy().reshape(NUM_ROWS_ALL - 1,
                                                                           NUM_COLS_ALL - 1)  # Transfer to CPU and convert to NumPy
                eyy_from_u = e_from_u[:, 1].detach().cpu().numpy().reshape(NUM_ROWS_ALL - 1,
                                                                           NUM_COLS_ALL - 1)  # Transfer to CPU and convert to NumPy
                rxy_from_u = e_from_u[:, 2].detach().cpu().numpy().reshape(NUM_ROWS_ALL - 1,
                                                                           NUM_COLS_ALL - 1)  # Transfer to CPU and convert to NumPy
            neptune_plot2d(exx_from_u, f"fit1/plot/exx_from_u", neptune=self.neptune)
            neptune_plot2d(eyy_from_u, f"fit1/plot/eyy_from_u", neptune=self.neptune)
            neptune_plot2d(rxy_from_u, f"fit1/plot/rxy_from_u", neptune=self.neptune)

        if STATE_MESSAGES:
            print("STATE: Fit-1 Finished.")

    def run_fit_2(self, num_epochs=NUM_FITTING_2_EPOCHS) -> None:
        if not FIT_DISPLACEMENT and FIT_STRAIN:
            optimizer = self.optimizer_class(
                list(self.strain_fit_model.parameters()),
                lr=LEARN_RATE
            )
        else:
            optimizer = self.optimizer_class(
                list(self.disp_fit_model.parameters()) +
                list(self.strain_fit_model.parameters()),
                lr=LEARN_RATE
            )

        if not FIT_DISPLACEMENT and FIT_STRAIN:
            self.strain_data = self.elas_model.create_coordinate_tensor(
                np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/strain_data'))

        if STATE_MESSAGES: print("STATE: Starting Fit 2 - Strain Fitting.")
        training_start_time = time.time()

        losses = []

        for e in range(num_epochs):
            print(f"Fit-2 Epoch {e} Starting.")
            epoch_start_time = time.time()

            self.disp_fit_model.train()
            self.strain_fit_model.train()
            for i in range(1, ITERATIONS_PER_EPOCH + 1):
                optimizer.zero_grad()

                if FIT_STRAIN_HIGH_RES_CORD:
                    disp_fit = self.disp_fit_model(self.disp_coord)
                    disp_fit_high = self.disp_fit_model(self.disp_coord_for_eval)
                    strain_fit = self.strain_fit_model(self.elas_coord_for_eval)
                else:
                    if not FIT_DISPLACEMENT and FIT_STRAIN:
                        strain_fit = self.strain_fit_model(self.strain_coord)
                    else:
                        disp_fit = self.disp_fit_model(self.disp_coord)
                        strain_fit = self.strain_fit_model(self.strain_coord)

                loss_list = None
                if i % SAVE_INTERVAL_LOSS_F2 == 0 or (e == 0 and i == 1):
                    loss_list = [e * ITERATIONS_PER_EPOCH + i]

                if FIT_STRAIN_HIGH_RES_CORD:
                    loss = self.fit_2_loss_highRES(
                        disp_fit=disp_fit,
                        disp_pred=disp_fit_high,
                        strain_fit=strain_fit,
                        loss_list=loss_list
                    )
                elif not FIT_DISPLACEMENT and FIT_STRAIN:
                    loss = self.discrepancy_loss(
                                fitted_data=strain_fit,
                                estimated_data= self.strain_data
                            )
                else:
                    loss = self.fit_2_loss(
                        disp_fit=disp_fit,
                        strain_fit=strain_fit,
                        loss_list=loss_list
                    )

                loss.backward()
                optimizer.step()

                if i % SAVE_INTERVAL_LOSS_F2 == 0 or (e == 0 and i == 1):
                    losses.append(loss_list)

                if i % NOTIFY_ITERATION_MOD == 0:
                    print(
                        f"Fit-2 Epoch: {e} [{i}/{ITERATIONS_PER_EPOCH} ({1.0 * i / ITERATIONS_PER_EPOCH * 100:.2f}%)]\tLoss: {loss.item():.6f}")

            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"Epoch{e} took {epoch_elapsed_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(num_epochs - e) * epoch_elapsed_time)}")

            self.neptune.append_value_to_field("fit2/epoch/loss", loss)

            self.disp_fit_model.save_eval(
                self.disp_coord_for_eval,
                path_ux=f"{OUTPUT_FOLDER}/pred_ux/fit2-{e}.txt",
                path_uy=f"{OUTPUT_FOLDER}/pred_uy/fit2-{e}.txt",
            )
            self.strain_fit_model.save_eval(
                self.strain_coord,
                path_exx=f"{OUTPUT_FOLDER}/pred_exx/fit2-{e}.txt",
                path_eyy=f"{OUTPUT_FOLDER}/pred_eyy/fit2-{e}.txt",
                path_rxy=f"{OUTPUT_FOLDER}/pred_rxy/fit2-{e}.txt",
            )

        u_pred = self.disp_fit_model(self.disp_coord_for_eval)
        if IS_DIM_PREDEFINED:
            dim = int(np.sqrt(self.disp_coord_for_eval[:, 0].shape[0]))
            ux_data = u_pred[:, 0].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
            uy_data = u_pred[:, 1].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
        else:
            ux_data = u_pred[:, 0].detach().cpu().numpy().reshape(NUM_ROWS_ALL,
                                                                  NUM_COLS_ALL)  # Transfer to CPU and convert to NumPy
            uy_data = u_pred[:, 1].detach().cpu().numpy().reshape(NUM_ROWS_ALL,
                                                                  NUM_COLS_ALL)  # Transfer to CPU and convert to NumPy
        if (SIMULATION_TYPE == "LOWRES"):
            ux_data = np.flip(ux_data, axis=1)
            uy_data = np.flip(uy_data, axis=1)
        neptune_plot2d(ux_data, f"fit2/plot/ux", neptune=self.neptune)
        neptune_plot2d(uy_data, f"fit2/plot/uy", neptune=self.neptune)

        e_pred = self.strain_fit_model(self.elas_coord_for_eval)
        if IS_DIM_PREDEFINED:
            dim = int(np.sqrt(self.elas_coord_for_eval[:, 0].shape[0]))
            exx_pred = e_pred[:, 0].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
            eyy_pred = e_pred[:, 1].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
            rxy_pred = e_pred[:, 2].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
        else:
            exx_pred = e_pred[:, 0].detach().cpu().numpy().reshape(NUM_ROWS_ALL - 1,
                                                                   NUM_COLS_ALL - 1)  # Transfer to CPU and convert to NumPy
            eyy_pred = e_pred[:, 1].detach().cpu().numpy().reshape(NUM_ROWS_ALL - 1,
                                                                   NUM_COLS_ALL - 1)  # Transfer to CPU and convert to NumPy
            rxy_pred = e_pred[:, 2].detach().cpu().numpy().reshape(NUM_ROWS_ALL - 1,
                                                                   NUM_COLS_ALL - 1)  # Transfer to CPU and convert to NumPy
        e_from_u = self.calc_strain(displacement=u_pred,disp_coord=self.disp_coord_for_eval,input_shape= torch.tensor([257, 257], device=DEVICE))
        neptune_plot2d(exx_pred, f"fit2/plot/exx", neptune=self.neptune)
        neptune_plot2d(eyy_pred, f"fit2/plot/eyy", neptune=self.neptune)
        neptune_plot2d(rxy_pred, f"fit2/plot/rxy", neptune=self.neptune)
        if IS_DIM_PREDEFINED:
            dim = int(np.sqrt(self.elas_coord_for_eval[:, 0].shape[0]))
            exx_from_u = e_from_u[:, 0].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
            eyy_from_u = e_from_u[:, 1].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
            rxy_from_u = e_from_u[:, 2].detach().cpu().numpy().reshape(dim, dim)  # Transfer to CPU and convert to NumPy
        else:
            exx_from_u = e_from_u[:, 0].detach().cpu().numpy().reshape(NUM_ROWS_ALL - 1,
                                                                   NUM_COLS_ALL - 1)  # Transfer to CPU and convert to NumPy
            eyy_from_u = e_from_u[:, 1].detach().cpu().numpy().reshape(NUM_ROWS_ALL - 1,
                                                                   NUM_COLS_ALL - 1)  # Transfer to CPU and convert to NumPy
            rxy_from_u = e_from_u[:, 2].detach().cpu().numpy().reshape(NUM_ROWS_ALL - 1,
                                                                   NUM_COLS_ALL - 1)  # Transfer to CPU and convert to NumPy
        neptune_plot2d(exx_from_u, f"fit2/plot/exx_from_u", neptune=self.neptune)
        neptune_plot2d(eyy_from_u, f"fit2/plot/eyy_from_u", neptune=self.neptune)
        neptune_plot2d(rxy_from_u, f"fit2/plot/rxy_from_u", neptune=self.neptune)
        if True:
            data_strain = np.loadtxt(f"../data/{DATAFOLDER}/{TRIAL_NAME}/strain_data")
            if IS_DIM_PREDEFINED:
                dim = int(np.sqrt(data_strain.shape[0]))
                data_exx = data_strain[:, 0].reshape(dim, dim)
                data_eyy = data_strain[:, 1].reshape(dim, dim)
                data_rxy = data_strain[:, 2].reshape(dim, dim)
            else:
                data_exx = data_strain[:, 0].reshape(NUM_ROWS_ALL, NUM_COLS_ALL)
                data_eyy = data_strain[:, 1].reshape(NUM_ROWS_ALL, NUM_COLS_ALL)
                data_rxy = data_strain[:, 2].reshape(NUM_ROWS_ALL, NUM_COLS_ALL)

            if (SIMULATION_TYPE == "LOWRES"):
                data_exx = np.flip(data_exx, axis=1)
                data_eyy = np.flip(data_eyy, axis=1)
                data_rxy = np.flip(data_rxy, axis=1)
            neptune_plot2d(data_exx, "true/plot/exx", neptune=self.neptune)
            neptune_plot2d(data_eyy, "true/plot/eyy", neptune=self.neptune)
            neptune_plot2d(data_rxy, "true/plot/rxy", neptune=self.neptune)
            neptune_plot2d(exx_pred-exx_from_u, f"fit2/plot/exx_pred_VS_fromU_error", neptune=self.neptune)
            neptune_plot2d(eyy_pred-eyy_from_u, f"fit2/plot/eyy_pred_VS_fromU_error", neptune=self.neptune)
            neptune_plot2d(rxy_pred-rxy_from_u, f"fit2/plot/rxy_pred_VS_fromU_error", neptune=self.neptune)
            neptune_plot2d(exx_pred-data_exx, f"fit2/plot/exx_pred_VS_true_error", neptune=self.neptune)
            neptune_plot2d(eyy_pred-data_eyy, f"fit2/plot/eyy_pred_VS_true_error", neptune=self.neptune)
            neptune_plot2d(rxy_pred-data_rxy, f"fit2/plot/rxy_pred_VS_true_error", neptune=self.neptune)
            neptune_plot2d(exx_from_u - data_exx, f"fit2/plot/exx_fromU_VS_true_error", neptune=self.neptune)
            neptune_plot2d(eyy_from_u - data_eyy, f"fit2/plot/eyy_fromU_VS_true_error", neptune=self.neptune)
            neptune_plot2d(rxy_from_u - data_rxy, f"fit2/plot/rxy_fromU_VS_true_error", neptune=self.neptune)

        if SAVE_FIT2_LOSS:
            np.savetxt(f"{OUTPUT_FOLDER}/fit2_loss.txt", losses)

        if STATE_MESSAGES: print("STATE: Fit-2 Finished.")

    def run_fit_3(self, num_epochs=NUM_FITTING_3_EPOCHS) -> None:
        optimizer = self.optimizer_class(self.intensity_fit_model.parameters(), lr=LEARN_RATE)

        if STATE_MESSAGES: print("STATE: Starting Fit 3 - Intensity Fitting.")
        training_start_time = time.time()

        for e in range(num_epochs):
            print(f"Fit-3 Epoch {e} Starting.")
            epoch_start_time = time.time()

            self.intensity_fit_model.train()
            for i in range(1, ITERATIONS_PER_EPOCH + 1):
                optimizer.zero_grad()
                int_fit = self.intensity_fit_model(self.strain_coord)
                loss_fit = self.fit_3_loss(int_fit=int_fit)
                error_fit = self.fit_3_error(int_fit=int_fit)
                if REGULARIZATION_L == 'L1':
                    l1_norm = sum(
                        p.abs().sum() for name, p in self.intensity_fit_model.named_parameters() if "weight" in name)
                    loss_reg = WEIGHT_L * l1_norm
                elif REGULARIZATION_L == 'L2':
                    l2_norm = sum(
                        p.pow(2).sum() for name, p in self.intensity_fit_model.named_parameters() if "weight" in name)
                    loss_reg = WEIGHT_L * l2_norm
                else:
                    loss_reg = 0
                loss = loss_fit + loss_reg

                loss.backward()
                optimizer.step()

                if i % NOTIFY_ITERATION_MOD == 0:
                    print(f"Fit-3 Epoch: {e} [{i}/{ITERATIONS_PER_EPOCH} ({1.0 * i / ITERATIONS_PER_EPOCH * 100:.2f}%)]\tLoss: {loss.item():.6f}")


            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"Epoch{e} took {epoch_elapsed_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(num_epochs - e) * epoch_elapsed_time)}")
            self.neptune.append_value_to_field("fit3/epoch/loss", loss)
            self.neptune.append_value_to_field("fit3/epoch/loss_reg", loss_reg)
            self.neptune.append_value_to_field("fit3/epoch/loss_fit", loss_fit)
            self.neptune.append_value_to_field("fit3/epoch/error_fit3", error_fit)
            # RUN["fit1/epoch/loss"].append(loss)

            self.intensity_fit_model.save_eval(
                self.elas_coord_for_eval,
                path_I=f"{OUTPUT_FOLDER}/pred_I/fit3-{e}.txt"
            )

        if IS_DIM_PREDEFINED:
            dim = int(np.sqrt(self.strain_coord[:, 0].shape[0]))
            int_pred = self.intensity_fit_model(self.strain_coord).detach().cpu().numpy().reshape(dim, dim)
            int_true = self.int_data.detach().cpu().numpy().reshape(dim, dim)
        else:
            int_pred = self.intensity_fit_model(self.strain_coord).detach().cpu().numpy().reshape(NUM_ROWS_ALL-1, NUM_COLS_ALL-1)
            int_true = self.int_data.detach().cpu().numpy().reshape(NUM_ROWS_ALL-1, NUM_COLS_ALL-1)

        dim = int(np.sqrt(self.strain_coord[:, 0].shape[0]))
        int_pred = self.intensity_fit_model(self.strain_coord).detach().cpu().numpy().reshape(dim, dim)
        int_true = self.int_data.detach().cpu().numpy().reshape(dim, dim)
        if REGULARIZATION_L == 'NONE':
            neptune_plot2d((int_pred), "true/plot/int_pred", title=f"Pred Int ", neptune=self.neptune)
        else:
            neptune_plot2d((int_pred), "true/plot/int_pred", title=f"Pred Int \n({REGULARIZATION_L} :W= {WEIGHT_L})", neptune=self.neptune)
        if not (SIMULATION_TYPE == "LOWRES"):
            neptune_plot2d((int_pred - int_true), "true/plot/int_error", neptune=self.neptune)

        if STATE_MESSAGES: print("STATE: Fit-3 Finished.")


    def run_train_elas(self, num_epochs=NUM_TRAINING_EPOCHS) -> None:
        if SEPARATED_ENu:
            optimizer = self.optimizer_class(
                list(self.disp_fit_model.parameters()) +
                list(self.strain_fit_model.parameters()) +
                list(self.elas_model.parameters()) +
                list(self.nu_model.parameters()),
                lr=LEARN_RATE
            )
        else:
            if FIT_STRAIN:
                optimizer = self.optimizer_class(
                    list(self.disp_fit_model.parameters()) +
                    list(self.strain_fit_model.parameters()) +
                    list(self.elas_model.parameters()),
                    lr=LEARN_RATE
                )
            else:
                optimizer = self.optimizer_class(
                    list(self.disp_fit_model.parameters()) +
                    list(self.elas_model.parameters()),
                    lr=LEARN_RATE
                )

        if STATE_MESSAGES: print("STATE: Starting Elas Model Training.")
        training_start_time = time.time()

        losses = []

        for e in range(num_epochs):
            print(f"Training Epoch {e} Starting.")
            epoch_start_time = time.time()

            if FIT_DISPLACEMENT:
                self.disp_fit_model.train()
            if FIT_STRAIN:
                self.strain_fit_model.train()
            if SEPARATED_ENu:
                self.nu_model.train()
            self.elas_model.train()
            for i in range(1, ITERATIONS_PER_EPOCH + 1):
                optimizer.zero_grad()

                if FIT_DISPLACEMENT:
                    disp_fit = self.disp_fit_model(self.disp_coord)
                    if FIT_STRAIN:
                        strain_fit = self.strain_fit_model(self.strain_coord)
                        strain_fit_elas_coord = self.strain_fit_model(self.elas_coord_for_eval.requires_grad_())
                    else:
                        disp_fit_strain_coord = self.disp_fit_model(self.disp_coord)
                        strain_fit = self.calc_strain(displacement= disp_fit_strain_coord,
                                                                disp_coord=self.disp_coord)
                        disp_fit_elas_coord = self.disp_fit_model(self.disp_coord_for_eval)
                        strain_fit_elas_coord = self.calc_strain(displacement= disp_fit_elas_coord,
                                                                disp_coord=self.disp_coord_for_eval)
                else:
                    disp_fit, strain_fit, strain_fit_elas_coord = None, None, None

                if SEPARATED_ENu:
                    nu_pred = self.nu_model(self.elas_coord)
                    E_pred = self.elas_model(self.elas_coord)
                    elas_pred = torch.cat(( E_pred,nu_pred), dim=1)
                else:
                    elas_pred = self.elas_model(self.elas_coord)

                loss_list = None
                if i % SAVE_INTERVAL_TRAIN == 0 or (e == 0 and i == 1):
                    loss_list = [e * ITERATIONS_PER_EPOCH + i]

                loss = self.train_elas_loss(
                    disp_fit=disp_fit,
                    strain_fit=strain_fit,
                    elas_pred=elas_pred,
                    strain_fit_elas_coord=strain_fit_elas_coord,
                    loss_list=loss_list
                )

                loss.backward()
                optimizer.step()

                if i % NOTIFY_ITERATION_MOD == 0:
                    print(f"Training Epoch: {e} [{i}/{ITERATIONS_PER_EPOCH} ({1.0 * i / ITERATIONS_PER_EPOCH * 100:.2f}%)]\tLoss: {loss.item():.6f}")

                if e == 0 and i == 1:
                    losses.append(loss_list)


                if i % SAVE_INTERVAL_TRAIN == 0:
                    if SEPARATED_ENu:
                        self.elas_model.save_eval(
                            self.elas_coord_for_eval,
                            path_E=f"{OUTPUT_FOLDER}/pred_E/train{e}-{i // SAVE_INTERVAL_TRAIN - 1}.txt",
                        )
                        self.nu_model.save_eval(
                            self.elas_coord_for_eval,
                            path_v=f"{OUTPUT_FOLDER}/pred_v/train{e}-{i // SAVE_INTERVAL_TRAIN - 1}.txt",
                        )
                    else:
                        self.elas_model.save_eval(
                            self.elas_coord_for_eval,
                            path_E=f"{OUTPUT_FOLDER}/pred_E/train{e}-{i // SAVE_INTERVAL_TRAIN - 1}.txt",
                            path_v=f"{OUTPUT_FOLDER}/pred_v/train{e}-{i // SAVE_INTERVAL_TRAIN - 1}.txt",
                        )
                    losses.append(loss_list)

            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"Epoch{e} took {epoch_elapsed_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(num_epochs - e) * epoch_elapsed_time)}")
            # RUN["train/epoch/loss"].append(loss)
            self.neptune.append_value_to_field("train/epoch/loss", loss)

            self.disp_fit_model.save_eval(
                self.disp_coord_for_eval,
                path_ux=f"{OUTPUT_FOLDER}/pred_ux/train{e}.txt",
                path_uy=f"{OUTPUT_FOLDER}/pred_uy/train{e}.txt",
            )
            if FIT_STRAIN:
                self.strain_fit_model.save_eval(
                    self.strain_coord,
                    path_exx=f"{OUTPUT_FOLDER}/pred_exx/train{e}.txt",
                    path_eyy=f"{OUTPUT_FOLDER}/pred_eyy/train{e}.txt",
                    path_rxy=f"{OUTPUT_FOLDER}/pred_rxy/train{e}.txt",
                )

            if e % SAVE_INTERVAL_TRAIN == 0:
                if SEPARATED_ENu:
                    last_nu = self.nu_model(self.elas_coord_for_eval)[:, 0]
                    last_elast = self.elas_model(self.elas_coord_for_eval)[:, 0]
                else:
                    last_elast = self.elas_model(self.elas_coord_for_eval)[:, 0]
                    last_nu = self.elas_model(self.elas_coord_for_eval)[:, 1]

                if FIT_STRAIN:
                    strain_fit_elas_coord = self.strain_fit_model(self.elas_coord_for_eval)
                else:
                    if DIFFERTIATION_METHOD == "AD":
                        strain_coord_grad = self.elas_coord_for_eval.requires_grad_()
                        disp_fit_elas_coord = self.disp_fit_model(strain_coord_grad)
                        strain_fit_elas_coord = self.calc_strain(displacement=disp_fit_elas_coord,
                                                                 disp_coord=strain_coord_grad)
                    else:
                        disp_fit_elas_coord = self.disp_fit_model(self.disp_coord_for_eval)
                        strain_fit_elas_coord = self.calc_strain.evalStrain(displacement=disp_fit_elas_coord)

                stress = self.calc_stress(
                    E=last_elast,
                    v=last_nu,
                    strain=strain_fit_elas_coord,
                )


                if IS_DIM_PREDEFINED:
                    S_xx = stress[:, 0]
                    X = self.elas_coord_for_eval[:, 0]
                    force_sim = CalForceBoundary(X, S_xx.unsqueeze(1))
                else:
                    S_yy = stress[:, 1]
                    X = self.elas_coord_for_eval[:, 1]
                    force_sim = CalForceBoundary(X, S_yy.unsqueeze(1))
                E_scaling = self.force_data/ force_sim

                if SIMULATION_TYPE == "LOWRES":
                    pred_e = torch.flip(last_elast.reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item()), dims=[1]).detach().cpu().numpy()
                    pred_e_scaled = E_scaling.detach().cpu().numpy() * pred_e
                    pred_nu = torch.flip(last_nu.reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item()), dims=[1]).detach().cpu().numpy()
                else:
                    pred_e = last_elast.reshape(ELAS_OUTPUT_SHAPE[0].item(),ELAS_OUTPUT_SHAPE[1].item()).detach().cpu().numpy()
                    pred_e_scaled = E_scaling.detach().cpu().numpy() * pred_e
                    pred_nu = last_nu.reshape(ELAS_OUTPUT_SHAPE[0].item(),ELAS_OUTPUT_SHAPE[1].item()).detach().cpu().numpy()

                # if not IS_DIM_PREDEFINED:
                #     pred_e = pred_e.T
                #     pred_e_scaled = pred_e_scaled.T
                #     pred_nu = pred_nu.T

                try:
                    MAE_E = np.mean(np.abs(pred_e - self.data_e))
                    MRE_E = np.mean(np.abs(pred_e - self.data_e)/self.data_e)
                    self.neptune.append_value_to_field("train/epoch/MAE_E", MAE_E)
                    self.neptune.append_value_to_field("train/epoch/MRE_E", MRE_E)
                    MAE_E_SC = np.mean(np.abs(pred_e_scaled - self.data_e))
                    MRE_E_SC = np.mean(np.abs(pred_e_scaled- self.data_e)/self.data_e)
                    self.neptune.append_value_to_field("train/epoch/MAE_ScaledE", MAE_E_SC)
                    self.neptune.append_value_to_field("train/epoch/MRE_ScaledE", MRE_E_SC)
                    MAE_nu = np.mean(np.abs(pred_nu - self.data_nu))
                    MRE_nu = np.mean(np.abs(pred_nu - self.data_nu)/self.data_nu)
                    self.neptune.append_value_to_field("train/epoch/MAE_nu", MAE_nu)
                    self.neptune.append_value_to_field("train/epoch/MRE_nu", MRE_nu)
                    neptune_plot2d(pred_e, f"train/epoch/plot/Elast_{e}", title=f"Epoch {e}\n({PREFIX}- MAE ={MAE_E})", neptune=self.neptune)
                    neptune_plot2d(pred_e_scaled, f"train/epoch/plot/ScaledElast_{e}", title=f"Epoch {e}\n({PREFIX}- MAE ={MAE_E_SC})", neptune=self.neptune)
                    neptune_plot2d(pred_nu, f"train/epoch/plot/Nu_{e}", title=f"Epoch {e}\n({PREFIX}- MAE = {MAE_nu})", neptune=self.neptune)

                except Exception as e:
                    warnings.warn(f"Unexpected error: {e}")

                # Calculate stress
                if FIT_STRAIN:
                    strain_fit_elas_coord = self.strain_fit_model(self.elas_coord_for_eval)
                else:
                    disp_fit_elas_coord = self.disp_fit_model(self.disp_coord_for_eval.requires_grad_())
                    strain_fit_elas_coord = self.calc_strain(displacement= disp_fit_elas_coord,
                                                            disp_coord=self.disp_coord_for_eval.requires_grad_())

                stress = self.calc_stress(
                    E=last_elast,
                    v=last_nu,
                    strain=strain_fit_elas_coord,
                )
                S_xx = stress[:, 0]
                S_yy = stress[:, 1]
                S_xy = stress[:, 2]

                makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Sxx/train{e}.txt")
                makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Syy/train{e}.txt")
                makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Sxy/train{e}.txt")
                np.savetxt(f"{OUTPUT_FOLDER}/pred_Sxx/train{e}.txt", S_xx.cpu().detach().numpy())
                np.savetxt(f"{OUTPUT_FOLDER}/pred_Syy/train{e}.txt", S_yy.cpu().detach().numpy())
                np.savetxt(f"{OUTPUT_FOLDER}/pred_Sxy/train{e}.txt", S_xy.cpu().detach().numpy())

            # if e == 100:
            #     visualize(self.neptune)

        neptune_plot2d(pred_e, f"train/epoch/plot/Elast_{num_epochs}", title=f"Epoch {num_epochs}\n({PREFIX})", neptune=self.neptune)
        if SAVE_TRAIN_LOSS:
            np.savetxt(f"{OUTPUT_FOLDER}/train_loss.txt", losses)

        # Log each loss to Neptune
        for tag, loss in zip(TAGS_LOSS, loss_list):
            self.neptune.set_value_to_field(tag, loss)
            # RUN[tag] = loss

        if STATE_MESSAGES: print("STATE: Fit-Elasticity Finished.")

    def run_train_elas_parametric(self, num_epochs=NUM_TRAINING_EPOCHS) -> None:
        optimizer = self.optimizer_class(
            list(self.disp_fit_model.parameters()) +
            list(self.strain_fit_model.parameters()) +
            list(self.intensity_fit_model.parameters()) +
            list(self.elas_model.parameters()),
            lr=LEARN_RATE
        )

        if STATE_MESSAGES: print("STATE: Starting Parametric Elas Model Training.")
        training_start_time = time.time()

        losses = []

        for e in range(num_epochs):
            print(f"Training Epoch {e} Starting.")
            epoch_start_time = time.time()

            if FIT_DISPLACEMENT:
                self.disp_fit_model.train()
                self.strain_fit_model.train()
            self.intensity_fit_model.train()
            self.elas_model.train()

            for i in range(1, ITERATIONS_PER_EPOCH + 1):
                optimizer.zero_grad()

                if FIT_DISPLACEMENT:
                    disp_fit = self.disp_fit_model(self.disp_coord_for_eval)
                    strain_fit = self.strain_fit_model(self.strain_coord)
                    strain_fit_elas_coord = self.strain_fit_model(self.elas_coord.requires_grad_())
                else:
                    disp_fit, strain_fit, strain_fit_elas_coord = None, None, None
                int_pred = self.intensity_fit_model(self.elas_coord)
                if FIXED_NU:
                    nu_value = self.elas_model.get_nu()
                    nu_pred = nu_value * torch.ones(int_pred.size()).to(DEVICE).squeeze(1)
                else:
                    nu_pred = self.elas_model(self.elas_coord).to(DEVICE).squeeze(1)
                if PARAMETRIC_NEED_a:
                    E_pred = (self.elas_model.get_a() * ((int_pred + self.elas_model.get_c()) ** self.elas_model.get_b())).squeeze(1)
                else:
                    E_pred = ((int_pred + self.elas_model.get_c()) ** self.elas_model.get_b()).squeeze(1)

                loss_list = None
                if i % SAVE_INTERVAL_TRAIN == 0 or (e == 0 and i == 1):
                    loss_list = [e * ITERATIONS_PER_EPOCH + i]



                loss = self.train_parametric_elas_loss(
                    disp_fit=disp_fit,
                    strain_fit=strain_fit,
                    E_pred=E_pred,
                    v_pred=nu_pred,
                    strain_fit_elas_coord=strain_fit_elas_coord,
                    loss_list=loss_list,
                )

                loss.backward()
                optimizer.step()

                if i % NOTIFY_ITERATION_MOD == 0:
                    print(f"Training Epoch: {e} [{i}/{ITERATIONS_PER_EPOCH} ({1.0 * i / ITERATIONS_PER_EPOCH * 100:.2f}%)]\tLoss: {loss.item():.6f}")

                if e == 0 and i == 1:
                    losses.append(loss_list)

                if i % SAVE_INTERVAL_TRAIN == 0:
                    int_pred_eval = self.intensity_fit_model(self.elas_coord_for_eval)
                    self.elas_model.save_eval(
                        self.elas_coord_for_eval,
                        int_pred_eval,
                        path_E=f"{OUTPUT_FOLDER}/pred_E/train{e}-{i // SAVE_INTERVAL_TRAIN - 1}.txt",
                        path_v=f"{OUTPUT_FOLDER}/pred_v/train{e}-{i // SAVE_INTERVAL_TRAIN - 1}.txt",
                    )
                    losses.append(loss_list)



            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"Epoch{e} took {epoch_elapsed_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(num_epochs - e) * epoch_elapsed_time)}")
            # RUN["train/epoch/loss"].append(loss)
            self.neptune.append_value_to_field("train/epoch/loss", loss)

            if e % SAVE_INTERVAL_TRAIN == 0:
                self.disp_fit_model.save_eval(
                    self.disp_coord_for_eval,
                    path_ux=f"{OUTPUT_FOLDER}/pred_ux/train{e}.txt",
                    path_uy=f"{OUTPUT_FOLDER}/pred_uy/train{e}.txt",
                )
                self.strain_fit_model.save_eval(
                    self.strain_coord,
                    path_exx=f"{OUTPUT_FOLDER}/pred_exx/train{e}.txt",
                    path_eyy=f"{OUTPUT_FOLDER}/pred_eyy/train{e}.txt",
                    path_rxy=f"{OUTPUT_FOLDER}/pred_rxy/train{e}.txt",
                )

                self.intensity_fit_model.save_eval(
                    self.elas_coord_for_eval,
                    path_I=f"{OUTPUT_FOLDER}/pred_I/train{e}.txt"
                )

                self.elas_model.save_eval(
                    self.elas_coord_for_eval,
                    int_pred_eval,
                    path_E=f"{OUTPUT_FOLDER}/pred_E/train{e}.txt",
                    path_v=f"{OUTPUT_FOLDER}/pred_v/train{e}.txt",
                )

                int_pred = self.intensity_fit_model(self.elas_coord_for_eval)
                if FIXED_NU:
                    nu_value = self.elas_model.get_nu()
                    last_nu = nu_value * torch.ones(int_pred.size())
                else:
                    last_nu = self.elas_model(self.elas_coord_for_eval)
                if PARAMETRIC_NEED_a:
                    last_elast = self.elas_model.get_a() * ((int_pred + self.elas_model.get_c()) ** self.elas_model.get_b())
                else:
                    last_elast = ((int_pred + self.elas_model.get_c()) ** self.elas_model.get_b())
                last_elast= last_elast.to(DEVICE).squeeze(1)
                last_nu= last_nu.to(DEVICE).squeeze(1)

                if FIT_STRAIN:
                    strain_fit_elas_coord = self.strain_fit_model(self.elas_coord_for_eval)
                else:
                    if DIFFERTIATION_METHOD == "AD":
                        strain_coord_grad = self.elas_coord_for_eval.requires_grad_()
                        disp_fit_elas_coord = self.disp_fit_model(strain_coord_grad)
                        strain_fit_elas_coord = self.calc_strain(displacement=disp_fit_elas_coord,
                                                                 disp_coord=strain_coord_grad)
                    else:
                        disp_fit_elas_coord = self.disp_fit_model(self.disp_coord_for_eval)
                        strain_fit_elas_coord = self.calc_strain.evalStrain(displacement=disp_fit_elas_coord)

                stress = self.calc_stress(
                    E=last_elast,
                    v=last_nu,
                    strain=strain_fit_elas_coord,
                )

                if IS_DIM_PREDEFINED:
                    S_xx = stress[:, 0]
                    X =  self.elas_coord_for_eval[:, 0]
                    force_sim = CalForceBoundary(X, S_xx.unsqueeze(1))
                else:
                    S_yy = stress[:, 1]
                    X =  self.elas_coord_for_eval[:, 1]
                    force_sim = CalForceBoundary(X, S_yy.unsqueeze(1))
                E_scaling = self.force_data/ force_sim

                if SIMULATION_TYPE == "LOWRES":
                    pred_e = torch.flip(last_elast.reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item()), dims=[1]).detach().cpu().numpy()
                    pred_e_scaled = E_scaling.detach().cpu().numpy() * pred_e
                    pred_nu = torch.flip(last_nu.reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item()), dims=[1]).detach().cpu().numpy()
                else:
                    pred_e = last_elast.reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item()).detach().cpu().numpy()
                    pred_e_scaled = E_scaling.detach().cpu().numpy() * pred_e
                    pred_nu = last_nu.reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item()).detach().cpu().numpy()

                try:
                    MAE_E = np.mean(np.abs(pred_e - self.data_e))
                    MRE_E = np.mean(np.abs(pred_e - self.data_e) / self.data_e)
                    self.neptune.append_value_to_field("train/epoch/MAE_E", MAE_E)
                    self.neptune.append_value_to_field("train/epoch/MRE_E", MRE_E)
                    MAE_E_SC = np.mean(np.abs(pred_e_scaled - self.data_e))
                    MRE_E_SC = np.mean(np.abs(pred_e_scaled - self.data_e) / self.data_e)
                    self.neptune.append_value_to_field("train/epoch/MAE_ScaledE", MAE_E_SC)
                    self.neptune.append_value_to_field("train/epoch/MRE_ScaledE", MRE_E_SC)
                    MAE_nu = np.mean(np.abs(pred_nu - self.data_nu))
                    MRE_nu = np.mean(np.abs(pred_nu - self.data_nu) / self.data_nu)
                    self.neptune.append_value_to_field("train/epoch/MAE_nu", MAE_nu)
                    self.neptune.append_value_to_field("train/epoch/MRE_nu", MRE_nu)
                    neptune_plot2d(pred_e, f"train/epoch/plot/Elast_{e}", title=f"Epoch {e}\n({PREFIX}- MAE ={MAE_E})",
                                   neptune=self.neptune)
                    neptune_plot2d(pred_e_scaled, f"train/epoch/plot/ScaledElast_{e}",
                                   title=f"Epoch {e}\n({PREFIX}- MAE ={MAE_E_SC})", neptune=self.neptune)
                    neptune_plot2d(pred_nu, f"train/epoch/plot/Nu_{e}", title=f"Epoch {e}\n({PREFIX}- MAE = {MAE_nu})",
                                   neptune=self.neptune)
                    print(f'Evaluation result : E MAE = {MAE_E:.4f} MRE ={MRE_E:.4f}')
                    print(f'Evaluation result : nu MAE = {MAE_E_SC:.4f} MRE ={MRE_E_SC:.4f}')
                    print(f'Evaluation result : E scale MAE = {MAE_nu:.4f} MRE ={MRE_nu:.4f}')
                except Exception as e:
                    warnings.warn(f"Unexpected error: {e}")

                # Calculate stress
                strain_fit_elas_coord = self.strain_fit_model(self.elas_coord_for_eval)
                stress = self.calc_stress(
                    E=last_elast,
                    v=last_nu,
                    strain=strain_fit_elas_coord,
                )
                S_xx = stress[:, 0]
                S_yy = stress[:, 1]
                S_xy = stress[:, 2]

                makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Sxx/train{e}.txt")
                makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Syy/train{e}.txt")
                makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Sxy/train{e}.txt")
                np.savetxt(f"{OUTPUT_FOLDER}/pred_Sxx/train{e}.txt", S_xx.cpu().detach().numpy())
                np.savetxt(f"{OUTPUT_FOLDER}/pred_Syy/train{e}.txt", S_yy.cpu().detach().numpy())
                np.savetxt(f"{OUTPUT_FOLDER}/pred_Sxy/train{e}.txt", S_xy.cpu().detach().numpy())

            # if e == 100:
            #     visualize(self.neptune)

        neptune_plot2d(pred_e, f"train/epoch/plot/Elast_{num_epochs}", title=f"Epoch {num_epochs}\n({PREFIX})", neptune=self.neptune)
        if SAVE_TRAIN_LOSS:
            np.savetxt(f"{OUTPUT_FOLDER}/train_loss.txt", losses)

        # Log each loss to Neptune
        for tag, loss in zip(TAGS_LOSS, loss_list):
            self.neptune.set_value_to_field(tag, loss)
            # RUN[tag] = loss

        if STATE_MESSAGES: print("STATE: Fit-Elasticity Finished.")

# ---------------------------------------------------------------------------- #

def makeDirIfNotExist(file_path):
    import os
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} is created')


import os
import re


def get_last_iteration_file(folder_path):
    # Define regex pattern to match filenames like train??-?.txt
    pattern = re.compile(r"train(\d+)-(\d)\.txt")

    last_epoch = -1
    last_iter = -1
    last_file = None

    # Iterate through all files in the directory
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))  # Extract epoch number
            iteration = int(match.group(2))  # Extract iteration number

            # Compare to find the file with the largest epoch and iteration
            if (epoch > last_epoch) or (epoch == last_epoch and iteration > last_iter):
                last_epoch = epoch
                last_iter = iteration
                last_file = filename

    if last_file:
        return os.path.abspath(os.path.join(folder_path, last_file))
    else:
        return None  # No matching file found


def get_n_rows_columns_coord(disp_coord_array):
    unique_x = np.unique(disp_coord_array[:, 0])  # Unique values in the first column (x-coordinates)
    unique_y = np.unique(disp_coord_array[:, 1])  # Unique values in the second column (y-coordinates)

    # Determine the number of rows and columns
    num_rows = len(unique_x)
    num_cols = len(unique_y)
    return num_rows, num_cols

def CalForceBoundary(X, stress):
    boundary_mask = (X == X.max())
    F = torch.sum(stress[boundary_mask, 0])
    return F
def CalculateForceFromDAT(DATAFOLDER,DATASETNAME):
    # Import data
    if IS_DIM_PREDEFINED:
        strain_coord_array = np.loadtxt(f'../data/{DATAFOLDER}/{DATASETNAME}/strain_coord')
        nu_array = np.loadtxt(f'../data/{DATAFOLDER}/{DATASETNAME}/nu_data')
        mu_array = np.loadtxt(f'../data/{DATAFOLDER}/{DATASETNAME}/m_data')
    else:
        strain_coord_array = np.loadtxt(f'../data/{DATAFOLDER}/{DATASETNAME}/disp_coord')
        nu_array = np.loadtxt(f'../data/{DATAFOLDER}/{DATASETNAME}/nu_data_nonc')
        mu_array = np.loadtxt(f'../data/{DATAFOLDER}/{DATASETNAME}/m_data_nonc')
    strain_data_array = np.loadtxt(f'../data/{DATAFOLDER}/{DATASETNAME}/strain_data')

    strain_coord = torch.tensor(strain_coord_array,device=DEVICE)
    strain_data = torch.tensor(strain_data_array,device=DEVICE)
    E = torch.tensor(mu_array,device=DEVICE)
    v = torch.tensor(nu_array,device=DEVICE)

    stress = CalculateStress()(E=E, v=v, strain=strain_data)

    if IS_DIM_PREDEFINED:
        X = strain_coord[:, 0]
        F = CalForceBoundary(X, stress[:, 0].unsqueeze(1))
    else:
        X = strain_coord[:, 1]
        F = CalForceBoundary(X, stress[:, 1].unsqueeze(1))
    return F


def main() -> None:
    # DEBUG: See where gradients are failing (the anomalies)
    # torch.autograd.set_detect_anomaly(True)
    global PREFIX, OUTPUT_FOLDER

    # LOAD Data. Called data_data sometimes as file is called _data.
    disp_coord_array = np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/{FILENAME_disp_coord}')
    disp_data_array = np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/{FILENAME_disp_data}')
    disp_coord_array_for_eval = np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/disp_coord')
    # disp_coord_array= np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/disp_coord')
    # disp_data_array= np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/disp_data_original')
    # m_data_array= np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/m_data')
    # nu_data_array= np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/nu_data')
    strain_coord_array = np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/{FILENAME_strain_coord}')

    elas_coord_array = np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/{FILENAME_elas_coord_array}')
    elas_coord_array_for_eval = np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/strain_coord')

    # strain_data_array= np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/strain_data')
    if not PARAMETRIC_E and SIMULATION_TYPE == "LOWRES":
        validate_n_rows_cols(disp_coord_array, strain_coord_array, elas_coord_array)

    data_e = np.loadtxt(f"../data/{DATAFOLDER}/{TRIAL_NAME}/m_data").reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())
    data_nu = np.loadtxt(f"../data/{DATAFOLDER}/{TRIAL_NAME}/nu_data").reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())

    if IS_MANUSCRIPT_RUN:
        if SIMULATION_TYPE == "NOISY":
            SIM_NAME = f'SNR_{SNR}'
            mm = np.mean(disp_data_array, 0)  # Calculate the mean of the entire array
            ss = np.abs(mm / SNR)  # Calculate ss based on mean and SNR
            print(f'Dataset with SNR{SNR} is generated : SS Ux= {ss[0]:.4e} , Uy= {ss[1]:.4e}')

            # Generate noise with shape matching displacement and standard deviation `ss`
            noise = np.random.normal(loc=0, scale=ss, size=disp_data_array.shape)
            print(f'Check generated noise : SS Ux= {np.std(noise, 0)[0]:.4e} , Uy= {np.std(noise, 0)[1]:.4e}')
            disp_data_array = disp_data_array + noise  # Add noise to displacement
            # np.savetxt(f'{saveFolder}/disp_data_SNR_{SNR}', y_disp)

    if STATE_MESSAGES: print("STATE: data imported")

    if DEVELOPMENT_MODE:
        neptune = None
        OUTPUT_FOLDER = OUTPUT_FOLDER + f'_TEST'
    else:
        neptune = Neptune()
        runID = neptune.run["sys/id"].fetch()
        OUTPUT_FOLDER = OUTPUT_FOLDER + f"_{runID}"

    logging_parameters(OUTPUT_FOLDER)

    inverse_problem = InverseProblem(
        disp_coordinates=disp_coord_array,
        disp_coord_array_for_eval=disp_coord_array_for_eval,
        strain_coordinates=strain_coord_array,
        disp_data=disp_data_array,
        elas_coordinates=elas_coord_array,
        elas_coord_array_for_eval=elas_coord_array_for_eval,
        data_e=data_e,
        data_nu = data_nu,
        neptune=neptune,
    )

    # RUN['model/architecture/name']=PREFIX

    if not DEVELOPMENT_MODE:
        neptune.set_value_to_field('model/architecture/name', PREFIX)
    # RUN['model/architecture/disp']=inverse_problem.disp_fit_model.architecture_description
    # RUN['model/architecture/strain']=inverse_problem.strain_fit_model.architecture_description
    # RUN['model/architecture/elast']=inverse_problem.elas_model.architecture_description

    # RUN["sys/tags"].add(f"{NUM_TRAINING_EPOCHS}_epochs")
    # RUN["sys/tags"].add(f"{ITERATIONS_PER_EPOCH}_iter_per_epoch")

    # # if not inverse_problem.load_pretrained_displacement():
    # inverse_problem.run_fit_1()
    # inverse_problem.save_displacement_model("fd_exp5_disp_f1")
    #
    # inverse_problem.run_fit_2()
    # inverse_problem.save_displacement_model("fd_exp5_disp_f2")
    # inverse_problem.save_strain_model("fd_exp5_strain_f2")
    #
    # inverse_problem.load_pretrained_displacement("fd_exp5_disp_f2")

    # Define model name
    if SIMULATION_TYPE == "GENERAL":
        disp_model_name = f'{DIFFERTIATION_METHOD}_{DATASETNAME}_disp_{MODEL_NAME}'
        strain_model_name = f'{DIFFERTIATION_METHOD}_{DATASETNAME}_strain_{MODEL_NAME}'
    elif SIMULATION_TYPE == "NOISY":
        disp_model_name = f'{DIFFERTIATION_METHOD}_{DATASETNAME}_disp_SNR_{SNR}_{MODEL_NAME}'
        strain_model_name = f'{DIFFERTIATION_METHOD}_{DATASETNAME}_strain_SNR_{SNR}_{MODEL_NAME}'
    elif SIMULATION_TYPE == "LOWRES":
        disp_model_name = f'{DIFFERTIATION_METHOD}_{DATASETNAME}_disp_LOWRES_n_{DOWNSAMPLING_N}_{MODEL_NAME}'
        strain_model_name = f'{DIFFERTIATION_METHOD}_{DATASETNAME}_strain_LOWRES_n_{DOWNSAMPLING_N}_{MODEL_NAME}'

    if WITH_PRETRAIN:
        if FIT_DISPLACEMENT:
            if FIT_STRAIN:
                if (not RETRAIN_DISPLACEMENT) and inverse_problem.load_pretrained_model_strain(f"{strain_model_name}_f2"):
                    print(f'Fit 2 has already been done: Loaded Model: {strain_model_name}_f2')
                else:
                    if (not RETRAIN_DISPLACEMENT) and inverse_problem.load_pretrained_model_displacement(f"{disp_model_name}_f1"):
                        print(f'Fit 1 has already been done: Loaded Model: {disp_model_name}_f1')
                    else:
                        print('Fit 1')
                        inverse_problem.run_fit_1()
                        inverse_problem.save_displacement_pretrained_model(f"{disp_model_name}_f1")
                        inverse_problem.save_displacement_model(f"{disp_model_name}_f1")

                    print('Fit 2')
                    inverse_problem.run_fit_2()
                    inverse_problem.save_displacement_pretrained_model(f"{disp_model_name}_f2")
                    inverse_problem.save_displacement_model(f"{disp_model_name}_f2")
                    inverse_problem.save_strain_pretrained_model(f"{strain_model_name}_f2")
                    inverse_problem.save_strain_model(f"{strain_model_name}_f2")
            else:
                if inverse_problem.load_pretrained_model_displacement(f"{disp_model_name}_f1"):
                    print(f'Fit 1 has already been done: Loaded Model: {disp_model_name}_f1')
                else:
                    print('Fit 1')
                    inverse_problem.run_fit_1()
                    inverse_problem.save_displacement_pretrained_model(f"{disp_model_name}_f1")
                    inverse_problem.save_displacement_model(f"{disp_model_name}_f1")


            if DOWNSAMPLING_N is not None and EVALUATE_ELAS_HIGH_RES:
                PREFIX = PREFIX + f"_ELAS_HIGH_RES"
                OUTPUT_FOLDER = ".././results"
                OUTPUT_FOLDER = OUTPUT_FOLDER + f"/{PREFIX}"
                # OUTPUT_FOLDER = OUTPUT_FOLDER + f"/{PREFIX}_rep2"

        elif FIT_STRAIN:
            print('Fit 2')
            inverse_problem.run_fit_2()
            inverse_problem.save_strain_pretrained_model(f"{strain_model_name}_f2")
            inverse_problem.save_strain_model(f"{strain_model_name}_f2")

        if PARAMETRIC_E:
            intensity_model_name = f'{DIFFERTIATION_METHOD}_{TRIAL_NAME}_int_f3'
            if not(RETRAIN_INTENSITY) and inverse_problem.load_pretrained_model_intensity(intensity_model_name):
                print(f'Fit 3 has already been done: Loaded Model: {intensity_model_name}')
            else:
                print('Fit 3')
                inverse_problem.run_fit_3()
                inverse_problem.save_intensity_pretrained_model(intensity_model_name)
                inverse_problem.save_intensity_model(intensity_model_name)

    # PREFIX = PREFIX + f"_rep2"
    # OUTPUT_FOLDER = ".././results"
    # OUTPUT_FOLDER = OUTPUT_FOLDER + f"/{PREFIX}"
    # # OUTPUT_FOLDER = OUTPUT_FOLDER + f"/{PREFIX}_rep2"

    if PARAMETRIC_E:
        inverse_problem.run_train_elas_parametric()
    else:
        inverse_problem.run_train_elas()
    inverse_problem.save_displacement_model(f"{disp_model_name}_t")
    if FIT_STRAIN:
        inverse_problem.save_strain_model(f"{strain_model_name}_t")
    if PARAMETRIC_E:
        inverse_problem.save_intensity_model(f"{intensity_model_name}_t")
    inverse_problem.save_elas_model(f"{DIFFERTIATION_METHOD}_{TRIAL_NAME}_elas_t")

    if not DEVELOPMENT_MODE:
        visualize(neptune)
    # plt.colorbar()

    if STATE_MESSAGES: print("STATE: Done")


def validate_n_rows_cols(disp_coord_array, strain_coord_array, elast_coord_array):
    global ELAS_OUTPUT_SHAPE, NUM_ROWS_ELAS, NUM_COLS_ELAS, NUM_ROWS_DISP, NUM_COLS_DISP, STRAIN_OUTPUT_SHAPE, DISP_OUTPUT_SHAPE
    num_rows, num_cols = get_n_rows_columns_coord(disp_coord_array)
    if num_rows != NUM_ROWS_DISP or num_cols != NUM_COLS_DISP:
        DISP_OUTPUT_SHAPE = torch.tensor([num_rows, num_cols], device=DEVICE)
        NUM_ROWS_DISP = num_rows
        NUM_COLS_DISP = num_cols

    num_rows, num_cols = get_n_rows_columns_coord(strain_coord_array)
    if num_rows != NUM_ROWS_ELAS or num_cols != NUM_COLS_ELAS:
        STRAIN_OUTPUT_SHAPE = torch.tensor([num_rows, num_cols], device=DEVICE)

    num_rows, num_cols = get_n_rows_columns_coord(elast_coord_array)
    if num_rows != NUM_ROWS_ELAS or num_cols != NUM_COLS_ELAS:
        ELAS_OUTPUT_SHAPE = torch.tensor([num_rows, num_cols], device=DEVICE)
        NUM_ROWS_ELAS = num_rows
        NUM_COLS_ELAS = num_cols


import matplotlib.pyplot as plt


def visualize(neptune=None):
    data_e = np.loadtxt(f"../data/{DATAFOLDER}/{TRIAL_NAME}/m_data").reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())
    data_v = np.loadtxt(f"../data/{DATAFOLDER}/{TRIAL_NAME}/nu_data").reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())
    neptune_plot2d(data_e, "true/plot/Elast", neptune = neptune)
    neptune_plot2d(data_v, "true/plot/nu", neptune = neptune)

    # disp_coord_array= np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/{FILENAME_disp_coord}')
    # disp_data_array= np.loadtxt(f'{PATH_TO_DATA}/{DATAFOLDER}/{TRIAL_NAME}/{FILENAME_disp_data}')
    data_disp = np.loadtxt(f"../data/{DATAFOLDER}/{TRIAL_NAME}/{FILENAME_disp_data}")
    if IS_DIM_PREDEFINED:
        dim = int(np.sqrt(data_disp.shape[0]))
        data_ux = data_disp[:, 0].reshape(dim, dim)
        data_uy = data_disp[:, 1].reshape(dim, dim)
    else:
        data_ux = data_disp[:, 0].reshape(NUM_ROWS_ALL,NUM_COLS_ALL)
        data_uy = data_disp[:, 1].reshape(NUM_ROWS_ALL, NUM_COLS_ALL)

    neptune_plot2d(data_ux, "true/plot/ux", neptune = neptune)
    neptune_plot2d(data_uy, "true/plot/uy", neptune = neptune)

    data_strain = np.loadtxt(f"../data/{DATAFOLDER}/{TRIAL_NAME}/strain_data")
    if IS_DIM_PREDEFINED:
        dim = int(np.sqrt(data_strain.shape[0]))
        data_exx = data_strain[:, 0].reshape(dim, dim)
        data_eyy = data_strain[:, 1].reshape(dim, dim)
        data_rxy = data_strain[:, 2].reshape(dim, dim)
    else:
        data_exx = data_strain[:, 0].reshape(NUM_ROWS_ALL,NUM_COLS_ALL)
        data_eyy = data_strain[:, 1].reshape(NUM_ROWS_ALL,NUM_COLS_ALL)
        data_rxy = data_strain[:, 2].reshape(NUM_ROWS_ALL,NUM_COLS_ALL)
    neptune_plot2d(data_exx, "true/plot/exx", neptune = neptune)
    neptune_plot2d(data_eyy, "true/plot/eyy", neptune = neptune)
    neptune_plot2d(data_rxy, "true/plot/rxy", neptune = neptune)

    last_elast = get_last_iteration_file(f"{OUTPUT_FOLDER}/pred_E")
    pred_e = np.loadtxt(last_elast).reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # im = ax.imshow(pred_e)
    neptune_plot2d(pred_e, "evaluation/plot/Elast", neptune = neptune)

    if IS_COMPRESSIBLE:
        last_v = get_last_iteration_file(f"{OUTPUT_FOLDER}/pred_v")
        pred_v = np.loadtxt(last_v).reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())
        neptune_plot2d(pred_v, "evaluation/plot/nu", neptune = neptune)

    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(pred_e - data_e),cmap=CMAP)  # 'tab20b'
    fig.colorbar(im, ax=ax)
    if neptune is not None:
        neptune.upload_to_field("evaluation/plot/error/Elast", fig)
    # RUN["evaluation/plot/error/Elast"].upload(fig)

    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(pred_v - data_v),cmap=CMAP)  # 'tab20b'
    fig.colorbar(im, ax=ax)
    if neptune is not None:
        neptune.upload_to_field("evaluation/plot/error/nu", fig)
    # RUN["evaluation/plot/error/nu"].upload(fig)

    # RUN["evaluation/error/absolute/Elast"] = np.mean(np.abs(pred_e - data_e))


def neptune_plot2d(data, str="evaluation/plot", title=None, neptune=None):
    fig, ax = plt.subplots()
    im = ax.imshow(data,cmap=CMAP)
    fig.colorbar(im, ax=ax)
    if title is not None:
        fig.suptitle(title)
    if neptune is not None:
        neptune.upload_to_field(str, fig)
    else:
        warnings.warn("neptune parameter is not provided")
    # RUN[str].upload(fig)
    plt.close(fig)  # Release memory


#Initialize simulation
# if IS_MANUSCRIPT_RUN:
#     print(f"TEST {DATASETNAME}")

main()
TIME_PER_BATCH = time.time() - BATCH_START_TIME
print("--- %s Elapsed time ---" % (TIME_PER_BATCH))
print('============================[:)]===================================')

variable_names = ['DATASETNAME', 'SIMULATION_TYPE', 'SIM_NAME', 'SNR', 'SIMULATION_OPTION', 'TIME_PER_BATCH']

parameters = {key: globals()[key] for key in variable_names if key in globals()}
makeDirIfNotExist(f"{OUTPUT_FOLDER}/parameters2.txt")
with open(f"{OUTPUT_FOLDER}/parameters2.txt", "w") as f:
    for key, value in parameters.items():
        f.write(f"{key}: {value}\n")

