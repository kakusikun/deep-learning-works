import os
import sys
from config.config_factory import _A as app_config
app_config.merge_from_file("/media/allen/mass/deep-learning-works/app.yml")
sys.path.insert(0, "/home/allen/R5/intel/computer_vision_sdk_2018.5.455/python/python3.6/ubuntu16")
from openvino.inference_engine import IENetwork, IEPlugin

model_xml = app_config.DNET.PATH
model_bin = os.path.splitext(model_xml)[0] + ".bin"
plugin = IEPlugin(device="CPU", plugin_dirs=None)
plugin.add_cpu_extension("/home/allen/R5/intel/computer_vision_sdk_2018.5.445/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_sse4.so")
# Read IR
print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
net = IENetwork(model=model_xml, weights=model_bin)
in_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
n, c, h, w = net.inputs[in_blob].shape
print(n, c, h, w)
DNet = plugin.load(network=net)
