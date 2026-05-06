import torch
import os

# Forza il riconoscimento dell'architettura se necessario
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

print(f"Versione PyTorch: {torch.__version__}")
print(f"ROCm/CUDA disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
