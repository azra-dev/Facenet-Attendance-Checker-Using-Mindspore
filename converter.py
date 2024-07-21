import tensorflow as tf
import numpy as np
import mindspore as ms
from mindspore import Tensor, save_checkpoint, load_checkpoint, load_param_into_net, nn
from mindspore.train.serialization import save_checkpoint
import os

class Converter():
    def __init__(self, 
                 input=None, 
                 output="models",
                 mindspore_model_class=None):
        self.input = input
        self.output = output
        self.mindspore_model_class = mindspore_model_class
    
    def load_pb_model(self, input):
        with tf.io.gfile.GFile(input, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.compat.v1.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph
    
    def extract_weights(self, graph):
        weights = {}
        with tf.compat.v1.Session(graph=graph) as sess:
            for var in tf.compat.v1.global_variables():
                weights[var.name] = sess.run(var)
        return weights

    def convert_weights(self, weights):
        ms_weights = {}
        for name, value in weights.items():
            # Assuming 'name' has a format compatible with MindSpore
            ms_weights[name] = Tensor(value)
        return ms_weights

    def pb_to_ckpt(self, input=None):
        img_name = os.path.basename(input)
        basename, ext = os.path.splitext(img_name)

        graph = self.load_pb_model(input)
        weights = self.extract_weights(graph)
        converted_weights = self.convert_weights(weights)

        # Ensure MindSpore model is defined
        if self.mindspore_model_class is None:
            raise ValueError("MindSpore model class must be provided.")

        # Create a MindSpore model instance
        model = self.mindspore_model_class()

        # Create a parameter dict from converted weights
        param_dict = {}
        for name, tensor in converted_weights.items():
            param_name = name.replace(':0', '')  # Adjust if needed
            param_dict[param_name] = tensor

        # Save checkpoint
        save_checkpoint(model, f"{self.output}/{basename}.ckpt")
        print(f"Model is successfully converted and saved: {self.output}/{basename}.ckpt")
            
def main():
    # Replace 'YourMindSporeModelClass' with your actual MindSpore model class
    from models.InceptionResNetV1 import InceptionResnetV1
    Conv = Converter(input="models/20180402-114759.pb", 
                     output="models",
                     mindspore_model_class=InceptionResnetV1)
    Conv.pb_to_ckpt("models/20180402-114759.pb")

if __name__ == "__main__":
    main()
