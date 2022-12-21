import tensorflow as tf

path = 'preprocessing/graph_v2.pb'

graph_def = tf.compat.v1.GraphDef()
loaded = graph_def.ParseFromString(open(path, 'rb').read())
init = tf.keras.initializers.GlorotNormal()


class extractor_v2(tf.keras.Model):
    def __init__(self):
        super(extractor_v2, self).__init__(name='')

        self.conv3_a = tf.keras.layers.Conv3D(16, 3, activation=tf.nn.relu, kernel_initializer=init, padding="same",
                                              name='conv3_a')
        self.conv3_b = tf.keras.layers.Conv3D(16, 3, activation=tf.nn.relu, kernel_initializer=init, padding="same",
                                              name='conv3_b')
        self.conv3_c = tf.keras.layers.Conv3D(32, 3, activation=tf.nn.relu, kernel_initializer=init, padding="same",
                                              name='conv3_c')
        self.conv3_d = tf.keras.layers.Conv3D(32, 3, activation=tf.nn.relu, kernel_initializer=init, padding="same",
                                              name='conv3_d')
        self.conv3_e = tf.keras.layers.Conv3D(64, 3, activation=tf.nn.relu, kernel_initializer=init, padding="same",
                                              name='conv3_e')
        self.conv3_f = tf.keras.layers.Conv3D(64, 3, activation=tf.nn.relu, kernel_initializer=init, padding="same",
                                              name='conv3_f')
        self.conv3_g = tf.keras.layers.Conv3D(64, 3, activation=tf.nn.relu, kernel_initializer=init, padding="same",
                                              name='conv3_g')
        self.conv3_h = tf.keras.layers.Conv3D(32, 3, activation=tf.nn.relu, kernel_initializer=init, padding="same",
                                              name='conv3_h')
        self.conv3_i = tf.keras.layers.Conv3D(16, 3, activation=tf.nn.relu, kernel_initializer=init, padding="same",
                                              name='conv3_i')
        self.conv3_j = tf.keras.layers.Conv3D(1, 1, kernel_initializer=init, padding="same", name='conv3_j')

        self.conv3_trans_a = tf.keras.layers.Conv3DTranspose(64, 3, strides=2, kernel_initializer=init, padding="same",
                                                             use_bias=False, name='conv3_transpose_a')
        self.conv3_trans_b = tf.keras.layers.Conv3DTranspose(32, 3, strides=2, kernel_initializer=init, padding="same",
                                                             use_bias=False, name='conv3_transpose_b')
        self.conv3_trans_c = tf.keras.layers.Conv3DTranspose(16, 3, strides=2, kernel_initializer=init, padding="same",
                                                             use_bias=False, name='conv3_transpose_c')

        self.maxpool_a = tf.keras.layers.MaxPool3D(strides=(2, 2, 2))
        self.maxpool_b = tf.keras.layers.MaxPool3D(strides=(2, 2, 2))
        self.maxpool_c = tf.keras.layers.MaxPool3D(strides=(2, 2, 2))

        self.dropout_a = tf.keras.layers.Dropout(0.3)
        self.dropout_b = tf.keras.layers.Dropout(0.3)
        self.dropout_c = tf.keras.layers.Dropout(0.3)
        self.dropout_d = tf.keras.layers.Dropout(0.3)
        self.dropout_e = tf.keras.layers.Dropout(0.3)
        self.dropout_f = tf.keras.layers.Dropout(0.3)

        self.concat = tf.keras.layers.Concatenate()

        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def call(self, input_tensor, training=False):
        x = self.conv3_a(input_tensor)
        conv1 = self.conv3_b(x)

        x = self.maxpool_a(conv1)
        x = self.dropout_a(x)

        x = self.conv3_c(x)
        conv2 = self.conv3_d(x)

        x = self.maxpool_b(conv2)
        x = self.dropout_b(x)

        x = self.conv3_e(x)
        conv3 = self.conv3_f(x)

        x = self.maxpool_c(conv3)
        x = self.dropout_c(x)

        x = self.conv3_trans_a(x)
        x = self.concat((x, conv3))
        x = self.conv3_g(x)

        x = self.dropout_d(x)

        x = self.conv3_trans_b(x)
        x = self.concat((x, conv2))
        x = self.conv3_h(x)

        x = self.dropout_e(x)

        x = self.conv3_trans_c(x)
        x = self.concat((x, conv1))
        x = self.conv3_i(x)

        x = self.dropout_e(x)

        x = self.conv3_j(x)

        output = self.sigmoid(x)

        return output


def return_weights(graph_def, layer_list):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    ret_list = []
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    for layer_name in layer_list:
        ret_list.append(tf.make_ndarray(
            tf.nest.map_structure(import_graph.as_graph_element, str(layer_name) + '/kernel').get_attr('value')))
        if layer_name.startswith('conv3d_transpose') == False:
            ret_list.append(tf.make_ndarray(
                tf.nest.map_structure(import_graph.as_graph_element, str(layer_name) + '/bias').get_attr('value')))

    return ret_list


layer_list = ['conv3d', 'conv3d_1', 'conv3d_2', 'conv3d_3', 'conv3d_4', 'conv3d_5', 'conv3d_transpose', 'conv3d_6',
              'conv3d_transpose_1', 'conv3d_7', 'conv3d_transpose_2', 'conv3d_8', 'conv3d_9']

weight_list = return_weights(
    graph_def, layer_list
)

extractor = extractor_v2()
extractor.build(input_shape=(None, 128, 128, 128, 1))
extractor.get_layer('conv3_a').set_weights(weight_list[:2])
extractor.get_layer('conv3_b').set_weights(weight_list[2:4])
extractor.get_layer('conv3_c').set_weights(weight_list[4:6])
extractor.get_layer('conv3_d').set_weights(weight_list[6:8])
extractor.get_layer('conv3_e').set_weights(weight_list[8:10])
extractor.get_layer('conv3_f').set_weights(weight_list[10:12])
extractor.get_layer('conv3_transpose_a').set_weights([weight_list[12]])
extractor.get_layer('conv3_g').set_weights(weight_list[13:15])
extractor.get_layer('conv3_transpose_b').set_weights([weight_list[15]])
extractor.get_layer('conv3_h').set_weights(weight_list[16:18])
extractor.get_layer('conv3_transpose_c').set_weights([weight_list[18]])
extractor.get_layer('conv3_i').set_weights(weight_list[19:21])
extractor.get_layer('conv3_j').set_weights(weight_list[21:23])