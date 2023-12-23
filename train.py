import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.legacy import Adam
from unet import build_unet
from metrics import dice_loss, dice_coef

# Global parameters
H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def custom_shuffle(a, b):
    combined = list(zip(a, b))
    np.random.shuffle(combined)
    return zip(*combined)

def custom_split(dataset, split_ratio=0.2):
    n = len(dataset)
    split_idx = int(n * split_ratio)
    return dataset[:-split_idx], dataset[-split_idx:]

def load_dataset(path, split=0.2):
    images = sorted(glob(os.path.join(path, "images", "*.png")))
    masks = sorted(glob(os.path.join(path, "masks", "*.png")))

    images, masks = custom_shuffle(images, masks)

    train_x, test_val_x = custom_split(images, split)
    train_y, test_val_y = custom_split(masks, split)

    valid_x, test_x = custom_split(test_val_x, 0.5)
    valid_y, test_y = custom_split(test_val_y, 0.5)

    # Convert file paths to byte strings for TensorFlow compatibility
    train_x = [str.encode(p) for p in train_x]
    train_y = [str.encode(p) for p in train_y]
    valid_x = [str.encode(p) for p in valid_x]
    valid_y = [str.encode(p) for p in valid_y]
    test_x = [str.encode(p) for p in test_x]
    test_y = [str.encode(p) for p in test_y]

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    #print(f"Image shape: {x.shape}")
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
   # print(f"Mask shape: {x.shape}")
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    print(f"Dataset X length: {len(X)}, Y length: {len(Y)}")
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

if __name__ == "__main__":
    # Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory for storing files
    create_dir("files")

    # Hyperparameters
    batch_size = 16
    lr = 1e-4
    num_epochs = 5
    print(num_epochs)
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")

    # Dataset
    dataset_path = "/Users/VladPavlovich/Downloads/BrainImagesMasks"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    # Model
    model = build_unet((H, W, 3))
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
