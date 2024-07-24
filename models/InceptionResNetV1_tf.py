import tensorflow as tf
from tensorflow.keras import layers

# Inception-Resnet-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu):
    tower_conv = layers.Conv2D(32, 1, activation='relu')(net)
    
    tower_conv1_0 = layers.Conv2D(32, 1, activation='relu')(net)
    tower_conv1_1 = layers.Conv2D(32, 3, padding='same', activation='relu')(tower_conv1_0)
    
    tower_conv2_0 = layers.Conv2D(32, 1, activation='relu')(net)
    tower_conv2_1 = layers.Conv2D(32, 3, padding='same', activation='relu')(tower_conv2_0)
    tower_conv2_2 = layers.Conv2D(32, 3, padding='same', activation='relu')(tower_conv2_1)
    
    mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], axis=3)
    up = layers.Conv2D(net.shape[-1], 1, activation=None)(mixed)
    
    net += scale * up
    if activation_fn:
        net = layers.Activation(activation_fn)(net)
    
    return net

# Inception-Resnet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu):
    tower_conv = layers.Conv2D(128, 1, activation='relu')(net)
    
    tower_conv1_0 = layers.Conv2D(128, 1, activation='relu')(net)
    tower_conv1_1 = layers.Conv2D(128, (1, 7), padding='same', activation='relu')(tower_conv1_0)
    tower_conv1_2 = layers.Conv2D(128, (7, 1), padding='same', activation='relu')(tower_conv1_1)
    
    mixed = tf.concat([tower_conv, tower_conv1_2], axis=3)
    up = layers.Conv2D(net.shape[-1], 1, activation=None)(mixed)
    
    net += scale * up
    if activation_fn:
        net = layers.Activation(activation_fn)(net)
    
    return net

# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu):
    tower_conv = layers.Conv2D(192, 1, activation='relu')(net)
    
    tower_conv1_0 = layers.Conv2D(192, 1, activation='relu')(net)
    tower_conv1_1 = layers.Conv2D(192, (1, 3), padding='same', activation='relu')(tower_conv1_0)
    tower_conv1_2 = layers.Conv2D(192, (3, 1), padding='same', activation='relu')(tower_conv1_1)
    
    mixed = tf.concat([tower_conv, tower_conv1_2], axis=3)
    up = layers.Conv2D(net.shape[-1], 1, activation=None)(mixed)
    
    net += scale * up
    if activation_fn:
        net = layers.Activation(activation_fn)(net)
    
    return net

# Reduction-A
def reduction_a(net, k, l, m, n):
    tower_conv = layers.Conv2D(n, 3, strides=2, padding='valid', activation='relu')(net)
    
    tower_conv1_0 = layers.Conv2D(k, 1, activation='relu')(net)
    tower_conv1_1 = layers.Conv2D(l, 3, padding='same', activation='relu')(tower_conv1_0)
    tower_conv1_2 = layers.Conv2D(m, 3, strides=2, padding='valid', activation='relu')(tower_conv1_1)
    
    tower_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(net)
    
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], axis=3)
    
    return net

# Reduction-B
def reduction_b(net):
    tower_conv = layers.Conv2D(256, 1, activation='relu')(net)
    tower_conv_1 = layers.Conv2D(384, 3, strides=2, padding='valid', activation='relu')(tower_conv)
    
    tower_conv1 = layers.Conv2D(256, 1, activation='relu')(net)
    tower_conv1_1 = layers.Conv2D(256, 3, strides=2, padding='valid', activation='relu')(tower_conv1)
    
    tower_conv2 = layers.Conv2D(256, 1, activation='relu')(net)
    tower_conv2_1 = layers.Conv2D(256, 3, padding='same', activation='relu')(tower_conv2)
    tower_conv2_2 = layers.Conv2D(256, 3, strides=2, padding='valid', activation='relu')(tower_conv2_1)
    
    tower_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(net)
    
    net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], axis=3)
    
    return net

# Inception ResNet V1 model
def inception_resnet_v1(inputs, is_training=True, dropout_keep_prob=0.8, bottleneck_layer_size=128):
    net = layers.Conv2D(32, 3, strides=2, padding='valid', activation='relu')(inputs)
    net = layers.Conv2D(32, 3, padding='valid', activation='relu')(net)
    net = layers.Conv2D(64, 3, padding='same', activation='relu')(net)
    net = layers.MaxPooling2D(3, strides=2, padding='valid')(net)
    net = layers.Conv2D(80, 1, padding='valid', activation='relu')(net)
    net = layers.Conv2D(192, 3, padding='valid', activation='relu')(net)
    net = layers.Conv2D(256, 3, strides=2, padding='valid', activation='relu')(net)
    
    # 5x Block35
    for _ in range(5):
        net = block35(net, scale=0.17)
    
    # Reduction-A
    net = reduction_a(net, 192, 192, 256, 384)
    
    # 10x Block17
    for _ in range(10):
        net = block17(net, scale=0.10)
    
    # Reduction-B
    net = reduction_b(net)
    
    # 5x Block8
    for _ in range(5):
        net = block8(net, scale=0.20)
    
    net = block8(net, activation_fn=None)
    
    net = layers.GlobalAveragePooling2D()(net)
    net = layers.Dropout(1 - dropout_keep_prob)(net)
    net = layers.Dense(bottleneck_layer_size, activation=None)(net)
    
    return net

if __name__ == '__main__':
    inputs = tf.keras.Input(shape=(160, 160, 3))
    net = inception_resnet_v1(inputs)
    model = tf.keras.Model(inputs, net)

    model.summary()
