from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from datetime import date
import tensorflow as tf
import numpy as np
import os

# === CONFIGURATION ===
Dataset_home_dir = "C:/Users/Rheiniel F. Damasco/Desktop/Paulit-ulit-na-katarantaduhan/Cleaned Dataset"  # 🖍️ Set this to your actual dataset
freeze_layers=True
Model_epoch = 2
resume_from_checkpoint = True  # 🔥 Toggle resume mode
checkpoint_dir = 'checkpoints_resnet/'  # 🔥 Checkpoint folder
os.makedirs(checkpoint_dir, exist_ok=True)

# === START TRAINING ===
def Resnet_train():
    model_title = "Resnet50V2(newgen_" + str(date.today()) + ")_" + str(Model_epoch) + "e.keras"

    train_dir = Dataset_home_dir + "/train"
    val_dir = Dataset_home_dir + "/test"
    img_size = 224

    # === Data Generators ===
    train_datagen = ImageDataGenerator(
        dtype='float32',
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        dtype='float32',
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
    )

    train_dataset = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        class_mode='categorical'
    )

    val_dataset = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        class_mode='categorical'
    )

    # === Class weights calculation ===
    class_labels = train_dataset.classes
    class_weight = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_labels),
        y=class_labels
    )
    class_weight_dict = dict(enumerate(class_weight))
    print("✅ Computed class weights:", class_weight_dict)

    # === Load or build model ===
    # latest_checkpoint = None
    # if resume_from_checkpoint:
    #     checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".keras")]
    #     if checkpoints:
    #         latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
    #         print(f"🚀 Resuming from checkpoint: {latest_checkpoint}")
    #         model = load_model(os.path.join(checkpoint_dir, latest_checkpoint))
    #     else:
    #         print("⚠️ No checkpoints found, starting from scratch.")
    #         latest_checkpoint = None

    # if not latest_checkpoint:
    #     base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    #     for layer in base_model.layers[-20:]:  # 🔥 Last 20 layers trainable
    #         layer.trainable = True

    #     x = base_model.output
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(1024, activation='relu')(x)
    #     predictions = Dense(train_dataset.num_classes, activation='softmax')(x)

    #     model = Model(inputs=base_model.input, outputs=predictions)
    #     print(model.summary())
    
    

    # === Callbacks ===
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,            # Wait for 5 epochs without improvement
        restore_best_weights=True,  # Restore the best weights
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'resnet_epoch_{epoch:02d}.keras'),
        save_weights_only=False,
        save_best_only=False,
        save_freq='epoch',
        verbose=1
    )
    # 🚀 Load your frozen model manually
    model_path = "checkpoints_resnet/resnet_epoch_01.keras"
    print(f"🚀 Loading frozen model from: {model_path}")
    model = load_model(model_path)

    # 🔥 Unfreeze ALL layers for fine-tuning
    for layer in model.layers:
        layer.trainable = True

    print("✅ Model layers unfrozen for fine-tuning.")
    # === Compile model ===
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # === Train! 🚀
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=Model_epoch,
        callbacks=[lr_scheduler, early_stopping, checkpoint],
    )

    # === Final save ===
    model.save(model_title)
    print(f"✅ Model saved to: {model_title}")

# Run
if __name__ == "__main__":
    Resnet_train()
