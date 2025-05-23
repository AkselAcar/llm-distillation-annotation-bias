from pathlib import Path
import torch


# Path-related constants ---------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"
CHECKPOINT_DIR = ROOT_DIR / "model_checkpoints"

if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
if not OUTPUTS_DIR.exists():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CFG_DIR = ROOT_DIR 



# Neural networks-related constants -----------------------------
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)

N_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
TEMPERATURE = 1.0
ALPHA = 0.5
CRITERION = "bce"
OPTIMIZER = "Adam"