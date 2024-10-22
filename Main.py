import os
import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load and check ONNX model
onnx_model_path = 'New_deep_model.onnx'
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))
print('ONNX model imported')

# Convert ONNX to TF
device = 'cpu'
tf_rep = prepare(onnx_model, device=device)
print("Preparation OK!")

# Export TF model
output_path = 'New_deep_model'
tf_rep.export_graph(output_path)
print("Tensorflow Export OK!")

# Load and analyze the saved model
model = tf.saved_model.load(output_path)

# Print input and output details
print("\nModel Information:")
print("Inputs:", model.signatures["serving_default"].inputs)
#print("Inputs:", list(model.signatures["serving_default"].inputs.keys()))
#print("Input Shapes:", [input.shape for input in model.signatures["serving_default"].inputs.values()])
print("Input Shapes:", [input_tensor.shape for input_tensor in model.signatures["serving_default"].inputs])

#print("Outputs:", list(model.signatures["serving_default"].outputs.keys()))
print("Outputs:", model.signatures["serving_default"].outputs)
print("Output Shapes:", [output_tensor.shape for output_tensor in model.signatures["serving_default"].outputs])
#print("Output Shapes:", [output.shape for output in model.signatures["serving_default"].outputs.values()])

# Print operations (optional)
print("\nModel Operations:")
for op in model.signatures["serving_default"].structured_outputs:
    print(op)
