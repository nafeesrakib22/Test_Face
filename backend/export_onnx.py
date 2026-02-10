import torch
import os
from backbones import get_model

# 1. Configuration
model_name = "edgeface_xs_gamma_06"
checkpoint_path = f"checkpoints/{model_name}.pt"
output_path = f"{model_name}.onnx"

# 2. Load the PyTorch Model
print(f"Loading {model_name}...")
device = torch.device('cpu')
model = get_model(model_name)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 3. Create Dummy Input
dummy_input = torch.randn(1, 3, 112, 112, requires_grad=True)

# 4. Export using Legacy Mode
print(f"Exporting to {output_path} (Legacy Mode)...")

# We use torch.onnx.export directly. 
# The previous error happened because newer PyTorch defaults to 'dynamo=True' internally for some paths.
# We will try to bypass it by not using 'export_params' in a complex way if possible,
# but usually, standard settings work if we don't trigger dynamo.

try:
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,          # Use Opset 13 (Stable for Quantization)
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
        verbose=False
    )
    print(f"Success! Model saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
except Exception as e:
    print("\n--- Export Failed ---")
    print(e)
    print("\nAttempting Fallback: Disabling dynamic axes...")
    # Fallback for strict quantization layers
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=['input'],
        output_names=['embedding']
    )
    print(f"Fallback Success! Model saved to {output_path}")