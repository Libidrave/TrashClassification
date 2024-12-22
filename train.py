from tensorflow.keras import backend
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Activation, Add, \
                                    Input, BatchNormalization, RandomRotation, RandomFlip, Rescaling
from tensorflow.keras.models import Model

import wandb
from wandb.integration.keras import WandbMetricsLogger
import os

run = wandb.init(
    project="trash-classification-nafis",
    config={
        "learning_rate": 0.001,
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 20,
        "batch_size" : 16
    }
)

config = wandb.config

def load_data(images_dir):
    # Without data augmentation
    train_ds = image_dataset_from_directory(
    images_dir,
    validation_split=0.3,
    subset="training",
    seed=42,
    image_size=(224, 224),
    interpolation="lanczos5",
    batch_size=config.batch_size,
    shuffle=True
    )

    val_ds = image_dataset_from_directory(
    images_dir,
    validation_split=0.3,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    interpolation="lanczos5",
    batch_size=config.batch_size
    )
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

# Create augmentation layer
def AugmentRescale(x):
  x = RandomRotation(0.2)(x)
  x = RandomFlip('horizontal')(x)
  x = Rescaling(1./127.5, offset=-1)(x)
  return x

# Create residual/skip connection layer
def CustomBlock(x, filter):
  x_copy = x

  x = Conv2D(filters = filter, kernel_size = (1,1), padding = "valid", use_bias=False)(x)
  x = BatchNormalization(axis=3, epsilon=1e-5)(x)
  x = Activation("mish")(x)

  x = Conv2D(filters = filter, kernel_size = (3,3), padding = "same", use_bias=False)(x)
  x = BatchNormalization(axis=3, epsilon=1e-5)(x)
  x = Activation("mish")(x)

  x_copy = Conv2D(filters = filter, kernel_size = (1,1), padding = "valid", use_bias=False)(x_copy)

  x = Add()([x, x_copy])
  x = Activation("mish")(x)
  return x

# Define CustomNet
def CustomNet(input_shape=(None, None, 3), num_classes=6, model_name="CustomNet"):
  inputs = Input(shape=input_shape) # Create Input layer
  x = AugmentRescale(inputs) # pass input layer through AugmentRescale layer

  # Create Conv -> MaxPooling layer before residual/skip connection layer
  x = Conv2D(filters=16, kernel_size=(7,7), padding="same")(x)
  x = BatchNormalization(axis=3, epsilon=1.001e-5)(x)
  x = Activation("mish")(x)
  x = MaxPooling2D(pool_size=(2,2))(x)

  # Pass the output from MaxPooling layer through residual/skip connection layer
  x = CustomBlock(x, 16)
  x = CustomBlock(x, 32)
  x = CustomBlock(x, 64)

  # Use GlobalAvgPooling to get (batch, filters) from last residual layer that can reduce model size but still carry the information from the feature
  # instead of using Flatten() which can cause model size explosion
  x = GlobalAveragePooling2D()(x)
  x = Dense(64, use_bias = False)(x)
  x = BatchNormalization(center = True, scale = False)(x)
  x = Activation('relu')(x)
  x = Dense(num_classes, activation="softmax", name="predictions")(x)

  model = Model(inputs=inputs, outputs=x, name=model_name)
  return model

def compile_train_model(train_ds, val_ds, model):
    model.compile(
    optimizer=Adam(learning_rate=config.learning_rate),
    loss=config.loss,
    metrics=[config.metric]
    )

    callback = ReduceLROnPlateau(
        monitor='val_loss',    # Use validation loss to adjust learning rate
        factor=0.5,            # Reduce learning rate by half
        patience=3,            # Wait for 3 epochs of no improvement
        min_lr=1e-6            # Minimum learning rate
    )
    
    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=config.epoch,
      callbacks=[callback, WandbMetricsLogger()]
    )

    # Save model locally
    model_path = "./models/models-trash-classification.keras"
    model.save(model_path)

    # Save model to W&B
    registered_name = "trash-classification-dev"

    run.link_model(path=model_path, registered_model_name=registered_name)

    print(f"The model has been saved in W&B")

    wandb.finish()

if __name__ == "__main__":

    images_dir = os.path.expanduser('dataset-resized')

    train_ds, val_ds = load_data(images_dir)

    model = CustomNet(input_shape=(224,224,3))

    compile_train_model(train_ds, val_ds, model)