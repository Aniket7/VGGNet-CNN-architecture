# import the necessary packages
import tensorflow as tf
class SmallVGGNetTF:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the input shape and channel dimension, assuming
		# TensorFlow/channels-last ordering
		inputShape = (height, width, depth)
		chanDim = -1

		# define the model input
		inputs = tf.keras.layers.Input(shape=inputShape)

		# first (CONV => RELU) * 2 => POOL layer set
		x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
		x = tf.keras.layers.Activation("relu")(x)
		x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
		x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
		x = tf.keras.layers.Activation("relu")(x)
		x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
		x = tf.keras.layers.Dropout(0.25)(x)

		# second (CONV => RELU) * 2 => POOL layer set
		x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
		x = tf.keras.layers.Activation("relu")(x)
		x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
		x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
		x = tf.keras.layers.Activation("relu")(x)
		x = tf.keras.layers.BatchNormalization(axis=chanDim)(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
		x = tf.keras.layers.Dropout(0.25)(x)

		# set of FC => RELU layers
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(512)(x)
		x = tf.keras.layers.Activation("relu")(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dropout(0.5)(x)

		# softmax classifier
		x = tf.keras.layers.Dense(classes)(x)
		x = tf.keras.layers.Activation("softmax")(x)

		# create the model
		model = tf.keras.models.Model(inputs, x, name="smallvggnet_tf")

		# return the constructed network architecture
		return model
