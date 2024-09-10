import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations


class CycleGAN():
    
    def __init__(self, imageHeight, imageWidth):
        # initialize the image height and width
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        
    def generator(self):
        # initialize the input layer
        inputs = layers.Input([self.imageHeight, self.imageWidth, 3])
        # weights initializer for the layers
        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=123)
        # c7s1-64
        g = layers.Conv2D(filters=64, kernel_size=7, strides=(1,1), padding='same', kernel_initializer=kernel_init)(inputs)
        g = layers.GroupNormalization(groups=-1)(g)
        g = activations.relu(g)
        # d128
        g = layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', kernel_initializer=kernel_init)(g)
        g = layers.GroupNormalization(groups=-1)(g)
        g = activations.relu(g)
        # d256
        g = layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same', kernel_initializer=kernel_init)(g)
        g = layers.GroupNormalization(groups=-1)(g)
        g = activations.relu(g)
        # R256
        if (self.imageHeight, self.imageWidth, 3) == (256, 256, 3):
            n_resnet = 9
        elif (self.imageHeight, self.imageWidth, 3) == (128, 128, 3):
            n_resnet = 6
        else:
            raise ValueError("Check image shape! Allowed only (256, 256, 3) or (128, 128, 3)")
        for _ in range(n_resnet):
            x = g
            g = layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', kernel_initializer=kernel_init)(g)
            g = layers.GroupNormalization(groups=-1)(g)
            g = activations.relu(g)
            # second convolutional layer
            g = layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', kernel_initializer=kernel_init)(g)
            g = layers.GroupNormalization(groups=-1)(g)
            # skip connection
            g = layers.Add()([g, x])
        # u128
        g = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer=kernel_init)(g)
        g = layers.GroupNormalization(groups=-1)(g)
        g = activations.relu(g)
        # u64
        g = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer=kernel_init)(g)
        g = layers.GroupNormalization(groups=-1)(g)
        g = activations.relu(g)
        # c7s1-3
        g = layers.Conv2D(filters=3, kernel_size=7, strides=(1,1), padding='same', kernel_initializer=kernel_init)(g)
        g = layers.GroupNormalization(groups=-1)(g)
        outputs = activations.tanh(g)
        
        # create the generator model
        generator = keras.models.Model(inputs=inputs, outputs=outputs)

        return generator

    def discriminator(self):
        # initialize input layer according to PatchGAN
        inputs = layers.Input(shape=[self.imageHeight, self.imageWidth, 3])
        # weights initializer for the layers
        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=321)
        # C64
        d = layers.Conv2D(filters=64, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=kernel_init)(inputs)
        d = layers.LeakyReLU(alpha=0.2)(d)
        # C128
        d = layers.Conv2D(filters=128, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=kernel_init)(d)
        d = layers.GroupNormalization(groups=-1)(d)
        d = layers.LeakyReLU(alpha=0.2)(d)
        # C256
        d = layers.Conv2D(filters=256, kernel_size=4, strides=(2,2), padding='same', kernel_initializer=kernel_init)(d)
        d = layers.GroupNormalization(groups=-1)(d)
        d = layers.LeakyReLU(alpha=0.2)(d)
        # C512
        d = layers.Conv2D(filters=512, kernel_size=4, strides=(1,1), padding='same', kernel_initializer=kernel_init)(d)
        d = layers.GroupNormalization(groups=-1)(d)
        d = layers.LeakyReLU(alpha=0.2)(d)
        # patch output
        outputs = layers.Conv2D(filters=1, kernel_size=4, strides=(1,1), padding='same', kernel_initializer=kernel_init)(d)
    
        # create the discriminator model
        discriminator = keras.models.Model(inputs=inputs, outputs=outputs)
        
        return discriminator