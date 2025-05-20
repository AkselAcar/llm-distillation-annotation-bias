from pathlib import Path
import torch


# Path-related constants ---------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"

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

N_EPOCHS = 1000
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
TEMPERATURE = 2.0
ALPHA = 0.5
CRITERION = "bce"
OPTIMIZER = "Adam"