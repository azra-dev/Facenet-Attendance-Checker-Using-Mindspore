import tensorflow as tf
import numpy as np
import mindspore as ms
from mindspore import Tensor, save_checkpoint
import os
import glob

class Converter():
    def __init__(self, 
                 input=None, 
                 output="models"):
        self.input = input
        self.output = output
    
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
        ms_weights = []
        for name, value in weights.items():
            ms_weights.append({
                "name": name,
                "data": Tensor(value)
            })
        return ms_weights

    def pb_to_ckpt(self, input=None):
        if input is None:
            if self.input is None:
                print("No input of .pb file is found, terminating the program.")
            else:
                input = self.input
        else:
            img_name = os.path.basename(input)
            basename, ext = os.path.splitext(img_name)

            graph = self.load_pb_model(input)
            weights = self.extract_weights(graph)
            converted_weights = self.convert_weights(weights)
            save_checkpoint(converted_weights, f"{self.output}/{basename}.ckpt")
            print(f"Model is successfully converted and saved: {self.output}/{basename}.ckpt")
            
def main():
    Conv = Converter()
    Conv.pb_to_ckpt("models/20180402-114759.pb")

if __name__ == "__main__":
    main()