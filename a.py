import mindspore as ms
import numpy as np
from mindspore import load_checkpoint
from models.InceptionResNetV1 import InceptionResnetV1

# def print_checkpoint(file_path):
#     param_dict = load_checkpoint(file_path)
    
#     if not param_dict:
#         print("The loaded parameter dict is empty. Please check the checkpoint file.")
#         return
    
#     for param_name, param_value in param_dict.items():
#         print(f"Parameter name: {param_name}, shape: {param_value.asnumpy().shape}")

# checkpoint_path = "models\\20180402-114759.ckpt"
# print_checkpoint(checkpoint_path)

def validate_model_with_input(model, input_shape):
    from mindspore import Tensor
    x = Tensor(np.random.randn(*input_shape).astype(np.float32))
    output = model(x)
    print("Output shape:", output.shape)

# Example usage
validate_model_with_input(InceptionResnetV1(), (1, 3, 160, 160))