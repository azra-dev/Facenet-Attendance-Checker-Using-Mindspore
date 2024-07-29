import tensorflow as tf
import os

def check_saved_model(pb_file_path):
    if not os.path.exists(pb_file_path):
        print(f"File does not exist: {pb_file_path}")
        return False
    
    try:
        # Try to load the SavedModel
        model = tf.saved_model.load(pb_file_path)
        print("Model loaded successfully.")
        
        # Print the model's signature to verify it's correctly loaded
        signatures = model.signatures
        for key in signatures:
            print(f"Signature: {key}")
            print(signatures[key])
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == '__main__':
    # Path to your .pb file or the directory containing the SavedModel
    model_path = 'saved_models/saved_model_e2'
    
    is_valid = check_saved_model(model_path)
    if is_valid:
        print("The .pb file is a valid SavedModel.")
    else:
        print("The .pb file is not a valid SavedModel.")
