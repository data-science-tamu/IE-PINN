import math
import warnings
import torch
import numpy as np
import time
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


TESTRUN = False



# Import config ===========================================================
TRIAL_NAME = config["trial"]["name"]
TRIAL_SAVEPATH = config["trial"]["savePath"]

GPU = config["device"]["GPU"]
CUDA_INDEX = config["device"]["CUDA_index"]

PRETRAIN = config["training"]["pretraining"]
NUM_FITTING_1_EPOCHS = config["training"]["epochs"]["fit1"]
NUM_FITTING_2_EPOCHS = config["training"]["epochs"]["fit2"]
NUM_TRAINING_EPOCHS = config["training"]["epochs"]["fit3"]
ITERATIONS_PER_EPOCH = config["training"]["iterationPerEpoch"]
LEARN_RATE = config["training"]["learning_rate"]
SAVE_INTERVAL_TRAIN = config["training"]["epochsPerModelSaving"]

NUM_HIDDEN_LAYER_FIT = config["model"]["fitting"]["hidden_layers"]
NUM_NEURON_FIT = config["model"]["fitting"]["neurons_per_layer"]
NUM_HIDDEN_LAYER_ELAS = config["model"]["elasticity"]["hidden_layers"]
NUM_NEURON_ELAS = config["model"]["elasticity"]["neurons_per_layer"]

     
DATASETPATH = config["data"]["dataset_path"]
DATASETNAME = config["data"]["dataset_name"]
NUM_ROWS_ELAS = config["data"]["elasticity"]["nRow"]
NUM_COLS_ELAS = config["data"]["elasticity"]["nCol"]
NUM_ROWS_DISP = config["data"]["displacement"]["nRow"]
NUM_COLS_DISP = config["data"]["displacement"]["nCol"]

SNR = config["noise"]["SNR"]
NOISETYPE = config["noise"]["type"]

LOADINGCOND = config["loading"]["condition"]
LOADINGFORCE =  config["loading"]["force"]

PE_INCLUDED = config["input"]["PE_included"]


WEIGHT_D_DISP = config["PINN"]["weight"]["displacement"]
WEIGHT_D_STRAIN = config["PINN"]["weight"]["strain"]
WEIGHT_R = config["PINN"]["weight"]["pde"]
WEIGHT_E = config["PINN"]["weight"]["meanE"]

E_CONSTRAINT = config["PINN"]["constrainedE"]



print(f'The dataset {DATASETNAME} is starting to train')

# Setting depened on config ==============================================
if NOISETYPE == "gaussian":
    isSpatialCor = False
elif NOISETYPE == "structured":
    isSpatialCor =  True
    PSF_sigma_l = 0.5
    PSF_sigma_a = 0.05
    PSF_f_c = 6e6

DATASET_PATH = f'{DATASETPATH}/{DATASETNAME}/{LOADINGCOND}/'


# ========================= Program Setup / Settings ========================= #
# ------------------------------ DEBUG SETTINGS ------------------------------ #
BATCH_START_TIME = time.time()
STATE_MESSAGES = True
NOTIFY_ITERATION_MOD = 100

# ----------------------------- GPU/CPU SETTING  -----------------------------#
# Selecting device to use. If use_cpu is true, will default
# to cpu, otherwise will use GPU if a GPU is available

if GPU:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
else:
    DEVICE = torch.device("cpu")

if CUDA_INDEX is not None :
    DEVICE = torch.device("cuda", index=CUDA_INDEX)

if STATE_MESSAGES: print("STATE: torch device is", DEVICE)


TENSOR_TYPE = torch.float32





# ------------------------------- TEST SETTING ------------------------------- #
if TESTRUN:
    CUDA_INDEX = 0

# ----------------------------- DATA DIMENSION SETTING ----------------------------- #
ELAS_INPUT_SHAPE = torch.tensor([NUM_ROWS_ELAS, NUM_COLS_ELAS], device=DEVICE)
DISP_INPUT_SHAPE = torch.tensor([NUM_ROWS_DISP, NUM_COLS_DISP], device=DEVICE)
DISP_OUTPUT_SHAPE = torch.tensor([NUM_ROWS_DISP, NUM_COLS_DISP], device=DEVICE)
STRAIN_INPUT_SHAPE = torch.tensor([NUM_ROWS_DISP, NUM_COLS_DISP], device=DEVICE)
ELAS_OUTPUT_SHAPE = torch.tensor([NUM_ROWS_ELAS, NUM_COLS_ELAS], device=DEVICE)
ELAS_EVAL_SHAPE = torch.tensor([NUM_ROWS_ELAS, NUM_COLS_ELAS], device=DEVICE)
STRAIN_OUTPUT_SHAPE = torch.tensor([NUM_ROWS_ELAS, NUM_COLS_ELAS], device=DEVICE)

STRAIN_SIZE = 3  # [ε_xx, ε_yy, γ_xy] (epsilon/e, epsilon/e, gamma/r) # Strain Fit out
DISPLACEMENT_SIZE = 2  # [u_x, u_y] # DispFit out
COORDINATE_SIZE = 2  # [x, y] # Model in
ELAS_SIZE = 2  # [pred_E, pred_v] # Elas out


# ----------------------------- MODEL PARAMETERS ----------------------------- #

class SirenAct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

# NUM_NEURON_FIT MUST BE EVEN (depth = NUM_NEURON // 2)
D_POS_ENC_FIT = NUM_NEURON_FIT // 2  # depth
assert (NUM_NEURON_FIT % 2 == 0)  # NOTE: ONLY FOR 2D INPUT DATA
ACTIVATION_FUNCTION_FIT = SirenAct  #torch.nn.SiLU # aka "swish",
ACTIVATION_FUNCTION_OUT_FIT = None

ACT_DISP = ACTIVATION_FUNCTION_FIT
ACT_DISP_OUT = ACTIVATION_FUNCTION_OUT_FIT
ACT_STRAIN = ACTIVATION_FUNCTION_FIT
ACT_STRAIN_OUT = ACTIVATION_FUNCTION_OUT_FIT

ACTIVATION_FUNCTION_ELAS = SirenAct  # torch.nn.ReLU
ACTIVATION_FUNCTION_OUT_ELAS = torch.nn.Softplus


if TESTRUN:
    NUM_HIDDEN_LAYER_FIT = 2
    NUM_HIDDEN_LAYER_ELAS = 2

    NUM_FITTING_1_EPOCHS = 1
    NUM_FITTING_2_EPOCHS = 1
    NUM_TRAINING_EPOCHS = 4
    ITERATIONS_PER_EPOCH = 10
    SAVE_INTERVAL_TRAIN = 2
    
# ------------------------------ OUTPUT FOLDERS ------------------------------ #
# Output Path. NOTE: Must already be present in file system relative to where
# script is ran.
if TRIAL_SAVEPATH is None:
    TRIAL_SAVEPATH = os.getcwd()
    print(TRIAL_SAVEPATH)
OUTPUT_FOLDER = f"{TRIAL_SAVEPATH}/results/{TRIAL_NAME}"
MODEL_SUBFOLDER = "/models"
PRE_FIT_MODEL_SUBFOLDER = "/pre_fitted"
SAVE_FIT2_LOSS = True  # FIT2: [i_net, ]
SAVE_TRAIN_LOSS = True  # TRAIN: [i_net, wld_d, wld_s, wle, wlr, total], # Weighted.

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
        pos_enc_x_cos = torch.cos(pos_enc_x[:, ::2])
        pos_enc_x_sin = torch.sin(pos_enc_x[:, 1::2])
        pos_enc_x = torch.concat([pos_enc_x_cos, pos_enc_x_sin], dim=1)

        pos_y = coord[:, 1]
        pos_enc_y = pos_y.reshape(-1, 1) * self.freq_tensor.reshape(1, -1)
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

        if PE_INCLUDED:
            self.pos_encode = PositionalEncoding2D(D_POS_ENC_FIT)
            self.hidden1 = torch.nn.Linear(self.pos_encode.dim_out, NUM_NEURON_FIT)
        else:
            self.hidden1 = torch.nn.Linear(2, NUM_NEURON_FIT)

        for i in range(2, self.num_layers):
            setattr(self, f"hidden{i}", torch.nn.Linear(NUM_NEURON_FIT, NUM_NEURON_FIT))
        self.out = torch.nn.Linear(NUM_NEURON_FIT, out_num)

        self.act1 = act_hidden()
        for i in range(2, self.num_layers):
            setattr(self, f"act{i}", act_hidden())


    def forward(self, x: torch.Tensor):
        if PE_INCLUDED:
            x = self.pos_encode(x)
        x = self.act1(self.hidden1(x))
        for i in range(2, self.num_layers):
            x = getattr(self, f"act{i}")(getattr(self, f"hidden{i}")(x))
        x = self.out(x)
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

    def save_eval(self, coordinates: torch.Tensor,path_ux: str, path_uy: str):
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
            path_exx: str, path_eyy: str, path_rxy: str
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

# ============================= Elasticity Model ============================= #
class ElasticityModel(torch.nn.Module):
    # [x, y] -> [E, v]
    def __init__(self):
        super().__init__()

        self.num_layers = NUM_HIDDEN_LAYER_ELAS

        if PE_INCLUDED:
            self.pos_encode = PositionalEncoding2D(D_POS_ENC_FIT)

            self.hidden1 = torch.nn.Linear(self.pos_encode.dim_out, NUM_NEURON_ELAS)
        else:
            self.hidden1 = torch.nn.Linear(2, NUM_NEURON_FIT)

        for i in range(2, self.num_layers):
            setattr(self, f"hidden{i}", torch.nn.Linear(NUM_NEURON_ELAS, NUM_NEURON_ELAS))
        self.out = torch.nn.Linear(NUM_NEURON_ELAS, ELAS_SIZE)

        self.act1 = ACTIVATION_FUNCTION_ELAS()
        for i in range(2, self.num_layers):
            setattr(self, f"act{i}", ACTIVATION_FUNCTION_ELAS())
        self.act_out = ACTIVATION_FUNCTION_OUT_ELAS()
        self.act_out2 = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        if PE_INCLUDED:
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
            device=DEVICE
        )

    def save_eval(
            self, coordinates: torch.Tensor,
            path_E: str, path_v: str
    ):
        self.eval()
        elas = self(coordinates)
        E = elas[:, 0]
        v = elas[:, 1]

        makeDirIfNotExist(path_E)
        makeDirIfNotExist(path_v)

        np.savetxt(path_E, E.cpu().detach().numpy())
        np.savetxt(path_v, v.cpu().detach().numpy())


# ---------------------------------------------------------------------------- #

# ==================== Loss Components and Helper Modules ==================== #
class DataLoss(torch.nn.Module):
    # loss_d, equation (10) - modified.
    # Mean of the absolute difference (fitted - actual).
    def __init__(self):
        super().__init__()

    def forward(self, fitted_data: torch.Tensor, actual_data: torch.Tensor):
        return torch.mean(torch.abs(fitted_data - actual_data))



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
        return torch.abs(torch.mean(pred_E) - E_CONSTRAINT)



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

    def forward(self, displacement: torch.Tensor, disp_coord: torch.Tensor):
        # Prepare the displacement values
        ux_mat = displacement[:, 0].reshape(1, 1, DISP_OUTPUT_SHAPE[0], DISP_OUTPUT_SHAPE[1])
        uy_mat = displacement[:, 1].reshape(1, 1, DISP_OUTPUT_SHAPE[0], DISP_OUTPUT_SHAPE[1])

        # Finite Differentiation using conv2d
        e_xx = self.conv_x(ux_mat)  # u_xx
        e_yy = self.conv_y(uy_mat)  # u_yy
        e_xy = self.conv_y(ux_mat) + self.conv_x(uy_mat)  # u_xy + u_yx

        # NOTE: From the paper the 100 constant
        e_xx = 100 * e_xx.reshape(-1)
        e_yy = 100 * e_yy.reshape(-1)
        e_xy = 100 * e_xy.reshape(-1)

        return torch.stack([e_xx, e_yy, e_xy], dim=1)

    def evalStrain(self, displacement: torch.Tensor):
        # Prepare the displacement values
        ux_mat = displacement[:, 0].reshape(1, 1, STRAIN_INPUT_SHAPE[0], STRAIN_INPUT_SHAPE[1])
        uy_mat = displacement[:, 1].reshape(1, 1, STRAIN_INPUT_SHAPE[0], STRAIN_INPUT_SHAPE[1])

        # Finite Differentiation using conv2d
        e_xx = self.conv_x(ux_mat)  # u_xx
        e_yy = self.conv_y(uy_mat)  # u_yy
        e_xy = self.conv_y(ux_mat) + self.conv_x(uy_mat)  # u_xy + u_yx

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
    def forward(self, pred_E: torch.Tensor, stress: torch.Tensor):
        # Prepare the stress values for self.conv2d
        sxx_mat = stress[:, 0].reshape(1, 1, ELAS_INPUT_SHAPE[0], ELAS_INPUT_SHAPE[1])
        syy_mat = stress[:, 1].reshape(1, 1, ELAS_INPUT_SHAPE[0], ELAS_INPUT_SHAPE[1])
        sxy_mat = stress[:, 2].reshape(1, 1, ELAS_INPUT_SHAPE[0], ELAS_INPUT_SHAPE[1])

        # Finite Differentiation using conv2d
        sxx_x = self.conv2d(sxx_mat, self.w_conv_x)
        syy_y = self.conv2d(syy_mat, self.w_conv_y)
        sxy_x = self.conv2d(sxy_mat, self.w_conv_x)
        sxy_y = self.conv2d(sxy_mat, self.w_conv_y)

        # Equilibrium Condition, equation (6).
        f_x = sxx_x + sxy_y
        f_y = syy_y + sxy_x

        # Normalize the losses, equation (8).
        E_hat_pred = self.calculate_E_hat(pred_E)
        f_x_norm = f_x / E_hat_pred
        f_y_norm = f_y / E_hat_pred


        # Loss of in each coordinate: L1
        loss_x = torch.mean(torch.abs(f_x_norm))
        loss_y = torch.mean(torch.abs(f_y_norm))


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
    def calculate_E_hat(self, pred_E: torch.Tensor) -> torch.Tensor:
        pred_E_matrix = torch.reshape(pred_E, [ELAS_INPUT_SHAPE[0], ELAS_INPUT_SHAPE[1]])
        pred_E_matrix_4d = torch.reshape(pred_E_matrix, [-1, 1, ELAS_INPUT_SHAPE[0], ELAS_INPUT_SHAPE[1]])
        pred_E_conv = self.conv2d(pred_E_matrix_4d, self.sum_kernel)
        return pred_E_conv

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
    ):
        
        # CALCULATE FORCE EXTERNAL
        if LOADINGFORCE is not None:
            self.force_data = LOADINGFORCE
            print(f'The calculated external force applied to boundary is {LOADINGFORCE} N')
        else:
            self.force_data = CalculateForceFromDAT(DATASET_PATH)
            print(f'The calculated external force applied to boundary is {self.force_data:.4f} N')

        # Initialize the component PyTorch Modules.
        # Models

        self.disp_fit_model = DisplacementFittingModel()
        self.disp_fit_model.to(DEVICE)
        self.disp_fit_model.apply(DisplacementFittingModel.init_weight_and_bias)

        self.strain_fit_model = StrainFittingModel()
        self.strain_fit_model.to(DEVICE)
        self.strain_fit_model.apply(StrainFittingModel.init_weight_and_bias)

        self.elas_model = ElasticityModel()
        self.elas_model.to(DEVICE)
        self.elas_model.apply(ElasticityModel.init_weight_and_bias)

        # For saving sake
        self.fit_1_epochs = 0

        # Various Components of Loss
        self.data_loss = DataLoss()
        self.discrepancy_loss = DiscrepancyLoss()
        self.elas_loss = ElasticityLoss()
        self.calc_strain = CalculateStrain()
        self.eq_loss = EquilibriumLoss()
        self.calc_stress = CalculateStress()

        # Optimizer depends on what kind of training
        self.optimizer_class = torch.optim.Adam

        # Create Coordinates. Copy in case the tensors would refer to the same memory.

        self.disp_coord = self.disp_fit_model.create_coordinate_tensor(disp_coordinates.copy())
        self.disp_coord_for_eval = self.elas_model.create_coordinate_tensor(disp_coord_array_for_eval.copy())
        self.strain_coord = self.strain_fit_model.create_coordinate_tensor(strain_coordinates.copy())
        self.elas_coord = self.elas_model.create_coordinate_tensor(elas_coordinates.copy())
        self.elas_coord_for_eval = self.elas_model.create_coordinate_tensor(elas_coord_array_for_eval.copy())

        # Fitting Data
        self.disp_data = torch.tensor(disp_data, dtype=TENSOR_TYPE, device=DEVICE)
        self.data_e = data_e
        self.data_nu = data_nu



    # --------------------------- Save Trained Models ---------------------------- #
    def save_displacement_model(self, file_path: str = f"disp") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.disp_fit_model.state_dict(), f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")

    def save_strain_model(self, file_path: str = "strain") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.strain_fit_model.state_dict(), f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")

    def save_elas_model(self, file_path: str = "elas") -> None:
        makeDirIfNotExist(f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")
        torch.save(self.elas_model.state_dict(), f"{OUTPUT_FOLDER}{MODEL_SUBFOLDER}/{file_path}.pt")






    # ---------------------- Error function for Each stage ----------------------- #
    def fit_1_error(self, disp_fit: torch.Tensor) -> torch.Tensor:
        return torch.mean((disp_fit - self.disp_data) ** 2)

    # ---------------------- Loss Functions for Each Stage ----------------------- #
    def fit_1_loss(self, disp_fit: torch.Tensor) -> torch.Tensor:
        data_loss = self.data_loss(
                fitted_data=disp_fit,
                actual_data=self.disp_data
            )
        total_data_loss = data_loss * WEIGHT_D_DISP
        return total_data_loss

    def fit_2_loss(
            self,
            disp_fit: torch.Tensor,
            disp_fit_strain: torch.Tensor,
            strain_fit: torch.Tensor,
            loss_list: list = None
    ) -> torch.Tensor:


        loss_d_strain = self.discrepancy_loss(
            fitted_data=strain_fit,
            estimated_data=self.calc_strain(displacement=disp_fit_strain,
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

    def train_elas_loss(
            self,
            disp_fit: torch.Tensor,
            disp_fit_strain: torch.Tensor,
            strain_fit: torch.Tensor,
            elas_pred: torch.Tensor,
            strain_fit_elas_coord: torch.Tensor,
            loss_list: list = None
    ) -> torch.Tensor:

       
     
        loss_d_disp = self.data_loss(
            fitted_data=disp_fit,
            actual_data=self.disp_data
        )

        loss_d_strain = self.discrepancy_loss(
            fitted_data=strain_fit,
            estimated_data=self.calc_strain(displacement=disp_fit_strain,
                                            disp_coord=self.disp_coord_for_eval)
        )

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
        loss_r = self.eq_loss(
            pred_E=E_pred,
            stress=stress
        )


        loss = loss_d_strain * WEIGHT_D_STRAIN + loss_d_disp * WEIGHT_D_DISP + loss_E * WEIGHT_E


        if loss_list != None:
            current_loss = [
                (loss_d_disp * WEIGHT_D_DISP).item(),
                (loss_d_strain * WEIGHT_D_STRAIN).item(),
                (loss_E * WEIGHT_E).item(),
                (loss_r * WEIGHT_R).item(),
                loss.item()
            ]

            loss_list += current_loss

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
                loss_fit = self.fit_1_loss(disp_fit = disp_fit)
                error_fit = self.fit_1_error(disp_fit=disp_fit)
                loss = loss_fit

                loss.backward()
                optimizer.step()

                if i % NOTIFY_ITERATION_MOD == 0:
                    print(f"Fit-1 Epoch: {e} [{i}/{ITERATIONS_PER_EPOCH} ({1.0 * i / ITERATIONS_PER_EPOCH * 100:.2f}%)]\tLoss: {loss.item():.6f}")

            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"Epoch{e} took {epoch_elapsed_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(num_epochs - e) * epoch_elapsed_time)}")

            self.disp_fit_model.save_eval(
                self.disp_coord,
                path_ux=f"{OUTPUT_FOLDER}/pred_ux/fit1-{e}.txt",
                path_uy=f"{OUTPUT_FOLDER}/pred_uy/fit1-{e}.txt",
            )
             
            self.fit_1_epochs += 1

        #Visualize U pred

        if STATE_MESSAGES:
            print("STATE: Fit-1 Finished.")
            print("=============================================")


    def run_fit_2(self, num_epochs=NUM_FITTING_2_EPOCHS) -> None:
        optimizer = self.optimizer_class(
                list(self.disp_fit_model.parameters()) +
                list(self.strain_fit_model.parameters()),
                lr=LEARN_RATE
            )

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

                disp_fit = self.disp_fit_model(self.disp_coord)
                disp_fit_strain = self.disp_fit_model(self.disp_coord)
                strain_fit = self.strain_fit_model(self.strain_coord)

                loss_list = None
                if i % 100 == 0 or (e == 0 and i == 1):
                    loss_list = [e * ITERATIONS_PER_EPOCH + i]

                loss = self.fit_2_loss(
                    disp_fit=disp_fit,
                    disp_fit_strain=disp_fit_strain,
                    strain_fit=strain_fit,
                    loss_list=loss_list
                )
                loss.backward()
                optimizer.step()

                if i % 100 == 0 or (e == 0 and i == 1):
                    losses.append(loss_list)

                if i % NOTIFY_ITERATION_MOD == 0:
                    print(f"Fit-2 Epoch: {e} [{i}/{ITERATIONS_PER_EPOCH} ({1.0 * i / ITERATIONS_PER_EPOCH * 100:.2f}%)]\tLoss: {loss.item():.6f}")

            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"Epoch{e} took {epoch_elapsed_time} seconds.")
            print(f"Elapsed program time is {timedelta(seconds=time.time() - training_start_time)}")
            print(f"Estimated time remaining is {timedelta(seconds=(num_epochs - e) * epoch_elapsed_time)}")

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

   
        if SAVE_FIT2_LOSS:
            np.savetxt(f"{OUTPUT_FOLDER}/fit2_loss.txt", losses)

        if STATE_MESSAGES: 
            print("STATE: Fit-2 Finished.")
            print("=============================================")



    def run_train_elas(self, num_epochs=NUM_TRAINING_EPOCHS) -> None:

        optimizer = self.optimizer_class(
            list(self.disp_fit_model.parameters()) +
            list(self.strain_fit_model.parameters()) +
            list(self.elas_model.parameters()),
            lr=LEARN_RATE
        )


        if STATE_MESSAGES: print("STATE: Starting Elas Model Training.")
        training_start_time = time.time()

        losses = []

        for e in range(num_epochs):
            print(f"Training Epoch {e} Starting.")
            epoch_start_time = time.time()



            self.disp_fit_model.train()
            self.strain_fit_model.train()

            self.elas_model.train()
            for i in range(1, ITERATIONS_PER_EPOCH + 1):
                optimizer.zero_grad()

                disp_fit = self.disp_fit_model(self.disp_coord)
                disp_fit_strain = self.disp_fit_model(self.disp_coord)
                
                strain_fit = self.strain_fit_model(self.strain_coord)
                strain_fit_elas_coord = self.strain_fit_model(self.elas_coord_for_eval.requires_grad_())
                    


                elas_pred = self.elas_model(self.elas_coord_for_eval)

                loss_list = None
                if i % SAVE_INTERVAL_TRAIN == 0 or (e == 0 and i == 1):
                    loss_list = [e * ITERATIONS_PER_EPOCH + i]

                loss = self.train_elas_loss(
                    disp_fit=disp_fit,
                    disp_fit_strain = disp_fit_strain,
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

            self.eval_train_elas(e = (e+1) )


        if SAVE_TRAIN_LOSS:
            np.savetxt(f"{OUTPUT_FOLDER}/train_loss.txt", losses)


        if STATE_MESSAGES: 
            print("STATE: Fit-Elasticity Finished.")
            print("=============================================")

    def eval_train_elas(self , e ):


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

        self.elas_model.save_eval(
            self.elas_coord_for_eval,
            path_E=f"{OUTPUT_FOLDER}/pred_E/train{e}.txt",
            path_v=f"{OUTPUT_FOLDER}/pred_v/train{e}.txt",
        )

        if e % SAVE_INTERVAL_TRAIN == 0 :

            last_elast = self.elas_model(self.elas_coord_for_eval)[:, 0]
            last_nu = self.elas_model(self.elas_coord_for_eval)[:, 1]
   
            strain_fit_elas_coord = self.strain_fit_model(self.elas_coord_for_eval)


            stress = self.calc_stress(
                E=last_elast,
                v=last_nu,
                strain=strain_fit_elas_coord,
            )

            makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Sxx/train{e}.txt")
            makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Syy/train{e}.txt")
            makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Sxy/train{e}.txt")
            np.savetxt(f"{OUTPUT_FOLDER}/pred_Sxx/train{e}.txt", stress[:, 0].cpu().detach().numpy())
            np.savetxt(f"{OUTPUT_FOLDER}/pred_Syy/train{e}.txt", stress[:, 1].cpu().detach().numpy())
            np.savetxt(f"{OUTPUT_FOLDER}/pred_Sxy/train{e}.txt", stress[:, 2].cpu().detach().numpy())


            X = self.elas_coord_for_eval[:, 0]
            S_xx = CalForceBoundary(X, stress[:, 0].unsqueeze(1))
            E_scaling = self.force_data / S_xx


            makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Elas/E_eval.txt")
            makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Elas/Nu_eval.txt")
            makeDirIfNotExist(f"{OUTPUT_FOLDER}/pred_Elas/E_scaled_eval.txt")
            np.savetxt(f"{OUTPUT_FOLDER}/pred_Elas/E_eval.txt", last_elast.cpu().detach().numpy())
            np.savetxt(f"{OUTPUT_FOLDER}/pred_Elas/E_scaled_eval.txt", (E_scaling * last_elast).cpu().detach().numpy())
            np.savetxt(f"{OUTPUT_FOLDER}/pred_Elas/Nu_eval.txt", last_nu.cpu().detach().numpy())


            pred_e = last_elast.reshape(ELAS_OUTPUT_SHAPE[0].item(), ELAS_OUTPUT_SHAPE[1].item()).detach().cpu().numpy()
            pred_e_scaled = E_scaling.detach().cpu().numpy() * pred_e
            pred_nu = last_nu.reshape(ELAS_OUTPUT_SHAPE[0].item(), ELAS_OUTPUT_SHAPE[1].item()).detach().cpu().numpy()

            try:
                MAE_E = np.mean(np.abs(pred_e - self.data_e))
                MAE_E_SC = np.mean(np.abs(pred_e_scaled - self.data_e))
                MAE_nu = np.mean(np.abs(pred_nu - self.data_nu))
                print(f"MAE: E = {MAE_E:.4f} >> Absolute E = {MAE_E_SC:.4f}  v = {MAE_nu:.4f}")

            except Exception as e:
                warnings.warn(f"Unexpected error: {e}")




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

def CalculateForceFromDAT(DATASET_PATH):
    # Import data
    strain_coord_array = np.loadtxt(f'{DATASET_PATH}/strain_coord')
    strain_data_array = np.loadtxt(f'{DATASET_PATH}/strain_data')
    nu_array = np.loadtxt(f'{DATASET_PATH}/nu_data')
    mu_array = np.loadtxt(f'{DATASET_PATH}/m_data')

    strain_coord = torch.tensor(strain_coord_array,device=DEVICE)
    strain_data = torch.tensor(strain_data_array,device=DEVICE)
    E = torch.tensor(mu_array,device=DEVICE)
    v = torch.tensor(nu_array,device=DEVICE)

    stress = CalculateStress()(E=E, v=v, strain=strain_data)

    X = strain_coord[:, 0]
    F = CalForceBoundary(X, stress[:, 0].unsqueeze(1))

    return F


def main() -> None:

    disp_coord_array = np.loadtxt(f'{DATASET_PATH}/disp_coord')
    disp_data_array = np.loadtxt(f'{DATASET_PATH}/disp_data')
    disp_coord_array_for_eval = np.loadtxt(f'{DATASET_PATH}/disp_coord')
    strain_coord_array = np.loadtxt(f'{DATASET_PATH}/strain_coord')
    elas_coord_array = np.loadtxt(f'{DATASET_PATH}/strain_coord')
    elas_coord_array_for_eval = np.loadtxt(f'{DATASET_PATH}/strain_coord')
    data_e = np.loadtxt(f"{DATASET_PATH}/m_data").reshape(ELAS_EVAL_SHAPE[0].item(),ELAS_EVAL_SHAPE[1].item())
    data_nu = np.loadtxt(f"{DATASET_PATH}/nu_data").reshape(ELAS_EVAL_SHAPE[0].item(),ELAS_EVAL_SHAPE[1].item())

    if SNR > 0:
        if isSpatialCor:
            # PSF filter ====================
            from scipy.signal import convolve2d
            l = np.arange(-128, 129)
            a = np.arange(-128, 129)
            L, A = np.meshgrid(l, a, indexing="xy")

            gaussian = np.exp(-(L**2 / PSF_sigma_l**2 + A**2 / PSF_sigma_a**2))

            modulation = np.cos(2 * np.pi * PSF_f_c * A)

            h = gaussian * modulation
            h /= np.sqrt(np.sum(h**2))

            R = np.zeros_like(disp_data_array, dtype=float)

            for c in range(2):
                # Reshape to 2D grid
                field = disp_data_array[:, c].reshape(257, 257)

                # Convolve with PSF
                Rc = convolve2d(field, h, mode="same")

                # Flatten back and store
                R[:, c] = Rc.reshape(-1)

            print(f'Done in convolution')
            #Generate noisy displacement data
            mm = np.mean(disp_data_array, 0)  # Calculate the mean of the entire array
            ss = np.abs(mm / SNR)  # Calculate ss based on mean and SNR
            noise = np.random.normal(loc=0, scale=ss, size=disp_data_array.shape)

            disp_data_array= R + noise
        else:
            #Generate noisy displacement data
            mm = np.mean(disp_data_array, 0)  # Calculate the mean of the entire array
            ss = np.abs(mm / SNR)  # Calculate ss based on mean and SNR
            print(f'Dataset with mean : mean Ux= {mm[0]:.4e} , Uy= {mm[1]:.4e}')
            print(f'Dataset with SNR{SNR} is generated : SS Ux= {ss[0]:.4e} , Uy= {ss[1]:.4e}')
            # Generate noise with shape matching displacement and standard deviation `ss`
            noise = np.random.normal(loc=0, scale=ss, size=disp_data_array.shape)
            print(f'Check generated noise : SS Ux= {np.std(noise, 0)[0]:.4e} , Uy= {np.std(noise, 0)[1]:.4e}')
            disp_data_array = disp_data_array + noise  # Add noise to displacement

    if STATE_MESSAGES: print("STATE: data imported")

    print(OUTPUT_FOLDER)

    logging_parameters(OUTPUT_FOLDER)

    #SAVE Displacement with noise data
    np.savetxt(f'{OUTPUT_FOLDER}/disp_data_SNR_{SNR}', disp_data_array)

    inverse_problem = InverseProblem(
        disp_coordinates=disp_coord_array,
        disp_coord_array_for_eval=disp_coord_array_for_eval,
        strain_coordinates=strain_coord_array,
        disp_data=disp_data_array,
        elas_coordinates=elas_coord_array,
        elas_coord_array_for_eval=elas_coord_array_for_eval,
        data_e=data_e,
        data_nu = data_nu
    )

    # Define model name
    disp_model_name = f'{DATASETNAME}_disp_SNR_{SNR}'
    strain_model_name = f'{DATASETNAME}_strain_SNR_{SNR}'


    if PRETRAIN:

        print('Fit 1')
        inverse_problem.run_fit_1()

        inverse_problem.save_displacement_model(f"{disp_model_name}_f1")

        print('Fit 2')
        inverse_problem.run_fit_2()

        inverse_problem.save_displacement_model(f"{disp_model_name}_f2")
        inverse_problem.save_strain_model(f"{strain_model_name}_f2")


    inverse_problem.run_train_elas()
    
    inverse_problem.save_displacement_model(f"{disp_model_name}_t")
    inverse_problem.save_strain_model(f"{strain_model_name}_t")
    inverse_problem.save_elas_model(f"{TRIAL_NAME}_elas_t")


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



def visualize_true(neptune=None):
    data_e = np.loadtxt(f"../data/compressible/{DATASETNAME}/m_data").reshape(ELAS_EVAL_SHAPE[0].item(),ELAS_EVAL_SHAPE[1].item())
    data_v = np.loadtxt(f"../data/compressible/{DATASETNAME}/nu_data").reshape(ELAS_EVAL_SHAPE[0].item(),ELAS_EVAL_SHAPE[1].item())
    neptune_plot2d(data_e, "true/plot/Elast", neptune = neptune,vmin=0.1,vmax=1.0)
    neptune_plot2d(data_v, "true/plot/nu", neptune = neptune,vmin=0.0,vmax=0.5)


    data_disp = np.loadtxt(f'{DATASET_PATH}/disp_data')

    dim = int(np.sqrt(data_disp.shape[0]))
    data_ux = data_disp[:, 0].reshape(dim, dim)
    data_uy = data_disp[:, 1].reshape(dim, dim)


    neptune_plot2d(data_ux, "true/plot/ux", neptune = neptune,vmin=0.0,vmax=2.5)
    neptune_plot2d(data_uy, "true/plot/uy", neptune = neptune,vmin=0.0,vmax=0.5)


    data_strain = np.loadtxt(f"../data/compressible/{DATASETNAME}/strain_data")

    dim = int(np.sqrt(data_strain.shape[0]))
    data_exx = data_strain[:, 0].reshape(dim, dim)
    data_eyy = data_strain[:, 1].reshape(dim, dim)
    data_rxy = data_strain[:, 2].reshape(dim, dim)

    neptune_plot2d(data_exx, "true/plot/exx", neptune = neptune,vmin=0.0,vmax=2.5)
    neptune_plot2d(data_eyy, "true/plot/eyy", neptune = neptune,vmin=-0.8,vmax=0.2)
    neptune_plot2d(data_rxy, "true/plot/rxy", neptune = neptune,vmin=-1.5,vmax=1.0)



def visualize(neptune=None):
    data_e = np.loadtxt(f"../data/compressible/{TRIAL_NAME}/m_data").reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())
    data_v = np.loadtxt(f"../data/compressible/{TRIAL_NAME}/nu_data").reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())
    neptune_plot2d(data_e, "true/plot/Elast", neptune = neptune)
    neptune_plot2d(data_v, "true/plot/nu", neptune = neptune)

    data_disp = np.loadtxt(f"../data/compressible/{TRIAL_NAME}/disp_data")

    dim = int(np.sqrt(data_disp.shape[0]))
    data_ux = data_disp[:, 0].reshape(dim, dim)
    data_uy = data_disp[:, 1].reshape(dim, dim)

    neptune_plot2d(data_ux, "true/plot/ux", neptune = neptune)
    neptune_plot2d(data_uy, "true/plot/uy", neptune = neptune)


    data_strain = np.loadtxt(f"../data/compressible/{TRIAL_NAME}/strain_data")

    dim = int(np.sqrt(data_strain.shape[0]))
    data_exx = data_strain[:, 0].reshape(dim, dim)
    data_eyy = data_strain[:, 1].reshape(dim, dim)
    data_rxy = data_strain[:, 2].reshape(dim, dim)

    neptune_plot2d(data_exx, "true/plot/exx", neptune = neptune)
    neptune_plot2d(data_eyy, "true/plot/eyy", neptune = neptune)
    neptune_plot2d(data_rxy, "true/plot/rxy", neptune = neptune)

    last_elast = get_last_iteration_file(f"{OUTPUT_FOLDER}/pred_E")
    pred_e = np.loadtxt(last_elast).reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())
    neptune_plot2d(pred_e, "evaluation/plot/Elast", neptune = neptune)

    last_v = get_last_iteration_file(f"{OUTPUT_FOLDER}/pred_v")
    pred_v = np.loadtxt(last_v).reshape(ELAS_INPUT_SHAPE[0].item(),ELAS_INPUT_SHAPE[1].item())
    neptune_plot2d(pred_v, "evaluation/plot/nu", neptune = neptune)

    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(pred_e - data_e),cmap='jet')  # 'tab20b'
    fig.colorbar(im, ax=ax)
    if neptune is not None:
        neptune.upload_to_field("evaluation/plot/error/Elast", fig)

    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(pred_v - data_v),cmap='jet')  # 'tab20b'
    fig.colorbar(im, ax=ax)
    if neptune is not None:
        neptune.upload_to_field("evaluation/plot/error/nu", fig)
    plt.close(fig)


def neptune_plot2d(data, str="evaluation/plot", title=None, neptune=None,vmin=None,vmax=None,coord=None):
    fig, ax = plt.subplots()
    if coord is not None:
        X = coord[:,0]
        Y = coord[:,1]
        if vmin is not None and vmax is not None:
            im = ax.scatter(X,Y,c=data,cmap='jet',vmin=vmin, vmax=vmax,s=100,marker='s')
        else:
            im = ax.scatter(X,Y,c=data,cmap='jet',s=100,marker='s')
    else:
        if vmin is not None and vmax is not None:
            im = ax.imshow(data,cmap='jet',vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(data,cmap='jet')
    fig.colorbar(im, ax=ax)
    if title is not None:
        fig.suptitle(title)
    if neptune is not None:
        neptune.upload_to_field(str, fig)
    else:
        warnings.warn("neptune parameter is not provided")
    # Release memory
    plt.close(fig)


main()

TIME_PER_BATCH = time.time() - BATCH_START_TIME
print("--- %s Elapsed time ---" % (TIME_PER_BATCH))
print('============================[:)]===================================')

variable_names = ['DATASETNAME', 'SNR', 'TIME_PER_BATCH']

parameters = {key: globals()[key] for key in variable_names if key in globals()}
makeDirIfNotExist(f"{OUTPUT_FOLDER}/parameters2-V2.txt")
with open(f"{OUTPUT_FOLDER}/parameters2-V2.txt", "w") as f:
    for key, value in parameters.items():
        f.write(f"{key}: {value}\n")

