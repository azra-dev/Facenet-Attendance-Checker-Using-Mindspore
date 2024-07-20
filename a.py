from mindspore.train.serialization import load_checkpoint

ckpt_file = 'models/20180402-114759.ckpt'
param_dict = load_checkpoint(ckpt_file)

if not param_dict:
    print("The checkpoint file is empty or the parameters could not be loaded.")
else:
    for param in param_dict:
        print(f"Parameter name: {param}, shape: {param_dict[param].shape}")