
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers

def create_pre_trained_model(local_weights_file):

  pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                  include_top = False,
                                  weights = None)

  pre_trained_model.load_weights(local_weights_file)

  # Make all the layers in the pre-trained model non-trainable
  for layer in pre_trained_model.layers:
    layer.trainable = False

  return pre_trained_model


def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=128, class_mode='binary', target_size=(150, 150))

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                  batch_size=32, class_mode='binary',
                                                                  target_size=(150, 150))

    return train_generator, validation_generator


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.93):
            print("\nReached 93% accuracy so cancelling training!")
            self.model.stop_training = True


def output_of_last_layer(pre_trained_model):
    last_desired_layer = pre_trained_model.get_layer('mixed9')
    print('last layer output shape: ', last_desired_layer.output_shape)
    last_output = last_desired_layer.output
    print('last layer output: ', last_output)
    return last_output


def create_final_model(pre_trained_model, last_output):
    from tensorflow.keras import Model
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=pre_trained_model.input, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def model_training(EPOCHS, pre_trained_model, model_checkpoint_path):
    import datetime

    start_time = datetime.datetime.now()
    train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

    callbacks = myCallback()
    last_output = output_of_last_layer(pre_trained_model)
    model = create_final_model(pre_trained_model, last_output)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor='loss', mode='min',
                                                          save_best_only=True)

    history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, verbose=2,
                        callbacks=[callbacks, model_checkpoint])
    end_time = datetime.datetime.now()
    print("Training time: ", end_time - start_time)
    return history


TRAINING_DIR = '/Computer vision, Time series, and NLP_TF certification/cats_vs_dogs/train_valid_split/training/'

VALIDATION_DIR = '/Computer vision, Time series, and NLP_TF certification/cats_vs_dogs/train_valid_split/testing/'

model_checkpoint_path = '/Computer vision, Time series, and NLP_TF certification/cats_vs_dogs/model_through_transfer_learning.h5'


pre_trained_model = create_pre_trained_model(local_weights_file)
history = model_training(10, pre_trained_model, model_checkpoint_path)

If you are going to submit the final trained model get it from this path: model_checkpoint_path




