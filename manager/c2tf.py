
import sys

try:
    sys.path.insert(0, "/home/allen/Documents/caffe/python")
    import caffe
    from caffe import layers as L
    from caffe import params as P
except ImportError:
    print("Need PyCaffe Dependency")
    pass

import numpy as np

import caffe.proto.caffe_pb2 as caffepb
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph


class Network():
    def __init__(self, deploy, caffemodel, name=None):
        
        net_params = caffepb.NetParameter()
        f = open(deploy, 'rb')
        net_str = f.read()
        text_format.Merge(net_str, net_params)

        self.weights = caffe.Net(deploy, caffemodel, caffe.TEST).params
        self.caffe_graph = net_params
        
        self.caffe_layer_list = {}
        self.set_caffe_layer_list()
        
        self.sess = tf.InteractiveSession()
        self.graph = self.sess.graph
        self.end_points = {}
        self.var_data_map = {}
        self.residual_block_idx = 1
        self.conv_block_idx = 0
        self.fc_idx = 0
        self.model_name = name if name is not None else ""
        self.scope = self.model_name
        
    def show_graph(self):
        for i in self.graph.as_graph_def().node:
            print("{:45} {:20}".format(i.name, i.op))
        
    def set_caffe_layer_list(self):
        for layer in self.caffe_graph.layer:
            self.caffe_layer_list[layer.name] = layer
        
    def get_layer(self, l_name, level='next'):
        next_layers = []
        if level == 'next':
            tops = self.caffe_layer_list[l_name].top
            for layer in self.caffe_graph.layer:
                for top_name in tops:
                    if top_name in layer.bottom:
#                         if self.caffe_layer_list[l_name].type == 'Convolution':
#                             if layer.type == 'ReLU' or layer.type == 'Convolution':
#                                 continue
                        next_layers.append(layer)
            if len(tops) > 1:
                return next_layers
            
        else:
            bottoms = self.caffe_layer_list[l_name].bottom
            for layer in self.caffe_graph.layer:
                for bottom_name in bottoms:
                    if bottom_name in layer.top:
#                         if self.caffe_layer_list[l_name].type == 'ReLU' or self.caffe_layer_list[l_name].type == 'Convolution':
#                             if layer.type == 'ReLU' or layer.type == 'Convolution':
#                                 continue
                        next_layers.append(layer)
            if len(bottoms) > 1:
                return next_layers
    
        if level == 'next':
            for i, layer in enumerate(next_layers):
                if l_name == layer.name:
                    return next_layers[i+1]
            return next_layers[0]
        else:
            for i, layer in enumerate(next_layers):
                if l_name == layer.name:
                    return next_layers[i-1]
            return next_layers[-1]
           
    
    def break_bn_scale(self):
        for layer in self.caffe_graph.layer:
            if layer.type == 'BatchNorm':
                layer.top[0] += "_bn"
            if layer.type == 'Scale':
                layer.bottom[0] += "_bn"
    
    def make_tf_variable(self, name, data, prefix=None):
        if prefix is not None:
            name = prefix + "/" + name
        self.var_data_map[name] = data
        return tf.compat.v1.get_variable(name, data.shape, trainable=False)
#         return tf.constant(value=weight, dtype=tf.float32)        
    
    def add_end_points(self, l, output):
        self.end_points[l.name] = output
               
        
    def add(self, l):   
        print(l.name)
        if l.type == 'Input':
            with tf.variable_scope(name_or_scope=self.scope):
                n, c, h, w = l.input_param.shape[0].dim
                shape = [n, h, w, c]
                output = tf.placeholder(tf.float32, shape=shape, name=l.name)
                self.add_end_points(l, output)
            
        elif l.type == 'BatchNorm':    
            scope = "{}/{}".format(self.scope, l.type)
            scale_layer = self.get_layer(l.name)            
            w1 = self.weights[l.name]
            raw_mean = w1[0].data
            raw_var = w1[1].data
            w2 = self.weights[scale_layer.name]
            raw_scale = w2[0].data
            raw_offset = w2[1].data
            
            bottom = self.get_layer(l.name, level='prev')
            inputs = self.end_points[bottom.name]
            input_shape = inputs.get_shape()
            mean = self.make_tf_variable(l.name+'_mean', raw_mean, scope)
            var = self.make_tf_variable(l.name+'_var', raw_var, scope)
            scale = self.make_tf_variable(l.name+'_scale', raw_scale, scope)
            offset = self.make_tf_variable(l.name+'_offset', raw_offset, scope)
            with tf.variable_scope(name_or_scope=scope):
                if len(input_shape) == 2:
                    inputs = tf.reshape(inputs, shape=[input_shape[0], 1, 1, input_shape[1]], name=l.name+"_expand")          
                    output = tf.nn.fused_batch_norm(x=inputs, scale=scale, offset=offset, mean=mean, variance=var, is_training=False, name=l.name)[0]
                    output = tf.reshape(output, shape=[input_shape[0], input_shape[1]], name=l.name+"_reduce")  
                    output = tf.identity(output, name='output')
                else:
                    output = tf.nn.fused_batch_norm(x=inputs, scale=scale, offset=offset, mean=mean, variance=var, is_training=False, name=l.name)[0]
                    output = tf.identity(output, name='output')
            self.add_end_points(scale_layer, output)
        
        elif l.type == 'Convolution':     
            self.conv_block_idx += 1
            self.scope = "{}/block{}/conv{}".format(self.model_name, self.residual_block_idx, self.conv_block_idx)
            params = l.convolution_param
            c_o = params.num_output
            bias = params.bias_term
            group = params.group
            p_h = params.pad_h
            p_w = params.pad_w
            k_h = params.kernel_h
            k_w = params.kernel_w
            s_h = params.stride_h
            s_w = params.stride_w
            dilation = params.dilation
            
            if s_h == 0 or s_w == 0:
                if len(params.stride) == 0:
                    s_h = s_w = 1
                else:
                    s_h = s_w = params.stride[0]
            if k_h == 0 or k_w == 0:
                k_h = k_w = params.kernel_size[0]
            if len(params.pad) > 0 and params.pad[0] > 0:
                p_h = p_w = params.pad[0]
                
            bottom = self.get_layer(l.name, level='prev')
            inputs = self.end_points[bottom.name]
            input_shape = inputs.get_shape()
            
            assert len(input_shape) == 4
            _, _, _, c_i = input_shape
            
            
                    
            raw_w = self.weights[l.name][0].data
            if group == 1:
                raw_w = raw_w.transpose((2, 3, 1, 0))                
                kernel = self.make_tf_variable(l.name+'_conv', raw_w, self.scope)
                with tf.variable_scope(name_or_scope=self.scope):
                    if p_h > 0:
                        inputs = tf.pad(inputs, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], name="{}_pad".format(l.name))
                        pad_type = 'VALID'
                    else:
                        if s_h > 1:
                            pad_type = 'SAME'
                        else:
                            pad_type = 'VALID'
                    output = tf.nn.conv2d(inputs, filter=kernel, strides=[1, s_h, s_w, 1], padding=pad_type, name=l.name)
                    output = tf.identity(output, name='output')
            elif group == c_i and group == c_o:                
                raw_w = raw_w.transpose((2, 3, 0, 1))
                kernel = self.make_tf_variable(l.name+'_conv', raw_w, self.scope)
                with tf.variable_scope(name_or_scope=self.scope):
                    if p_h > 0:
                        inputs = tf.pad(inputs, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], name="{}_pad".format(l.name))
                        pad_type = 'VALID'
                    else:
                        if s_h > 1:
                            pad_type = 'SAME'
                        else:
                            pad_type = 'VALID'
                    output = tf.nn.depthwise_conv2d(inputs, filter=kernel, strides=[1,s_h, s_w,1], rate=[1,1], padding=pad_type, name=l.name)
                    output = tf.identity(output, name='output')
            else:
                raw_w = raw_w.transpose((2, 3, 1, 0))
                kernel = self.make_tf_variable(l.name+'_conv', raw_w, self.scope)
                with tf.variable_scope(name_or_scope=self.scope):
                    if p_h > 0:
                        inputs = tf.pad(inputs, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], name="{}_pad".format(l.name))
                        pad_type = 'VALID'
                    else:
                        if s_h > 1:
                            pad_type = 'SAME'
                        else:
                            pad_type = 'VALID'
                    # Split the input into groups and then convolve each of them independently
                    input_groups = tf.split(axis=3, num_or_size_splits=group, value=inputs)
                    kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
                    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=pad_type)
                    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                    # Concatenate the groups
                    output = tf.compat.v1.concat(values=output_groups, axis=3)
                    output = tf.identity(output, name='output')
                
            if bias:
                raw_b = self.weights[l.name][1].data
                biases = self.make_tf_variable(l.name+'_bias', raw_b, self.scope)
                with tf.variable_scope(name_or_scope=self.scope):
                    output = tf.nn.bias_add(output, biases, name=l.name+"_bias")        
                    output = tf.identity(output, name='output')
            self.add_end_points(l, output)
        
        elif l.type == 'ReLU' or l.type == 'PReLU':
            scope = "{}/{}".format(self.scope, l.type)
            bottom = self.get_layer(l.name, level='prev')
            inputs = self.end_points[bottom.name]
            with tf.variable_scope(name_or_scope=scope):
                output = tf.nn.relu(inputs, name=l.name)
                output = tf.identity(output, name='output')
            self.add_end_points(l, output)
        
#         elif l.type == 'PReLU':
#             bottom = self.get_layer(l.name, level='prev')[-1]
#             inputs = self.end_points[bottom.name]
#             raw_alpha = self.weights[l.name][0].data
#             alpha = self.make_tf_variable(l.name+"_alpha", raw_alpha)  
#             with tf.compat.v1.variable_scope(l.name, values=[inputs]) as scope:
#                 output = tf.nn.relu(inputs) + tf.multiply(alpha, (inputs - tf.abs(inputs))) * 0.5
#             self.add_end_points(l, output)
        
        elif l.type == 'InnerProduct':
            self.fc_idx += 1
            self.scope = "{}/fc{}".format(self.model_name, self.fc_idx)
            params = l.inner_product_param
            c_o = params.num_output
            bias = params.bias_term
            
            bottom = self.get_layer(l.name, level='prev')
            inputs = self.end_points[bottom.name]
            input_shape = inputs.get_shape()            
            
            raw_w = self.weights[l.name][0].data
            raw_w = raw_w.transpose((1,0))
            if bias:
                raw_b = self.weights[l.name][1].data
            else:
                raw_b = np.zeros(c_o)
                
            weights = self.make_tf_variable(l.name+'_fc_w', raw_w, self.scope)
            biases = self.make_tf_variable(l.name+'_fc_b', raw_b, self.scope)
            with tf.variable_scope(name_or_scope=self.scope):
                if len(input_shape) == 4:
                    n, h, w, c = input_shape 
                    c_i = h*w*c
                    inputs = tf.reshape(inputs, shape=[1, c_i])
                else:
                    n, c_i = input_shape
                output = tf.compat.v1.nn.xw_plus_b(x=inputs, weights=weights, biases=biases, name=l.name)
                output = tf.identity(output, name='output')
            self.add_end_points(l, output)
            
        elif l.type == 'Scale':
            print("Merge into BN")
            
        elif l.type == 'Eltwise':
            self.residual_block_idx += 1
            self.conv_block_idx = 0
            self.scope = "{}/block{}".format(self.model_name, self.residual_block_idx)
            bottoms = self.get_layer(l.name, level='prev')
            output = None
            op = l.eltwise_param.operation
            if op == 1:
                with tf.variable_scope(name_or_scope=self.scope):
                    for i, bottom in enumerate(bottoms):
                        if bottom.type == 'BatchNorm':
                            continue
                        if output is None:
                            output = self.end_points[bottom.name]
                        else:
                            output += self.end_points[bottom.name]
                            output = tf.identity(output, name='output')
            else:
                print("Unsupported layer {}/{} at {}".format(l.type, op, l.name))
                sys.exit(1)
            self.add_end_points(l, output)
        elif l.type == 'Pooling':
            scope = "{}/{}".format(self.scope, l.type)
            bottom = self.get_layer(l.name, level='prev')
            inputs = self.end_points[bottom.name]
            
            params = l.pooling_param
            p_h = params.pad_h
            p_w = params.pad_w
            k_size = params.kernel_size
            if params.global_pooling:
                input_shape = inputs.get_shape()
                k_size = input_shape[1]
            s = params.stride           
            op = params.pool   
            '''
            enum PoolMethod {
                MAX = 0;
                AVE = 1;
                STOCHASTIC = 2;
              }
            '''
            with tf.variable_scope(name_or_scope=scope):
                if p_h > 0:
                    inputs = tf.pad(inputs, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], name="{}_pad".format(l.name))
                    pad_type = 'VALID'
                else:
                    if s > 1:
                        pad_type = 'SAME'
                    else:
                        pad_type = 'VALID'

                if op == 0:
                    output = tf.nn.max_pool2d(inputs, ksize=[1, k_size, k_size, 1], strides=[1, s, s, 1], padding=pad_type, name=l.name)
                if op == 1:
                    output = tf.nn.avg_pool2d(inputs, ksize=[1, k_size, k_size, 1], strides=[1, s, s, 1], padding=pad_type, name=l.name)
                
                output = tf.identity(output, name='output')
                
            self.add_end_points(l, output)
            
        elif l.type == 'Sigmoid':
            scope = "{}/{}".format(self.scope, l.type)
            bottom = self.get_layer(l.name, level='prev')
            inputs = self.end_points[bottom.name]
            input_shape = inputs.get_shape()
            with tf.variable_scope(name_or_scope=scope):
                if len(input_shape) == 4 and input_shape[1] == input_shape[2] == 1:
                    inputs = tf.reshape(inputs, shape=[input_shape[0], input_shape[3]], name=l.name+"_flatten")
                output = tf.nn.sigmoid(inputs, name=l.name)
                output = tf.identity(output, name='output')
            self.add_end_points(l, output)
            
        elif l.type == 'Flatten':
            axis = l.flatten_param.axis
            assert axis == 1
            scope = "{}/{}".format(self.scope, l.type)
            bottom = self.get_layer(l.name, level='prev')
            inputs = self.end_points[bottom.name]
            input_shape = inputs.get_shape()
            with tf.variable_scope(name_or_scope=scope):
                output = tf.reshape(inputs, shape=[input_shape[0], input_shape[1]*input_shape[2]*input_shape[3]], name=l.name)
                output = tf.identity(output, name='output')
            self.add_end_points(l, output)
            
        else:
            print("Unsupported layer {} at {}".format(l.type, l.name))
            sys.exit(1)
            
    def build(self, flag=None):
        for layer in self.caffe_graph.layer:
            self.add(layer)
            if flag is not None:
                if layer.name == flag:
                    break
    
    def allocate_weights(self):
        for name in self.var_data_map.keys():
            with tf.compat.v1.variable_scope("", reuse=True):
                data = self.var_data_map[name]
                var = tf.compat.v1.get_variable(name)
                self.sess.run(var.assign(data))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Use command:\n")
        print("python c2tf.py $PROTOTXT $CAFFEMODEL $OUTPUT_DIR $OUTPUT_MODEL_NAME $OUTPUT_NODE_NAME")
        sys.exit(1)    

    deploy = sys.argv[1]
    caffemodel = sys.argv[2]
    name = sys.argv[3]
    path = sys.argv[4]
    output_name = sys.argv[5]
    prefix = "{}/{}".format(path, name)

    assert "prototxt" in deploy
    assert "caffemodel" in caffemodel
    
    network = Network(deploy, caffemodel, name=name)
    network.build()
    network.allocate_weights()
    train_writer = tf.summary.FileWriter(path)
    train_writer.add_graph(network.graph)

    saver = tf.compat.v1.train.Saver()
    saver.save(network.sess, "{}.ckpt".format(prefix))

    with tf.io.gfile.GFile('{}.pb'.format(prefix), 'wb') as f:
        f.write(network.sess.graph_def.SerializeToString())

    d = freeze_graph(".pb".format(prefix), "",
                    True, "{}.ckpt".format(prefix),
                    output_name, 'save/restore_all',
                    'save/Const:0', "{}/frozen_{}.pb".format(path, name),
                    True, '')

