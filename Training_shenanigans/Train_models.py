from ultralytics import YOLO
import tensorflow as tf
from datetime import date
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
"""
    This file main purpose is to make training both YOLO and ResNet502V be easy to users.
    Training time varies depending on the hardware used, especially when GPU is involved.
    
    Training YOLO or ResNet502V at 10 epochs will be finished on about 30 minutes to 1 hour at most.
    The hardware used is I5 - 13500HX, GTX 4050 4GB. 
    
    IMPORTANT!!
        Be sure that you will not be using your computer in the near future
        as training requires the whole resources of the computer being used!
    
    Below are the requirements to be able to start training.
    
        - Dataset location for ResNet50V2
        - YAML file for YOLO
        - Epoch Size
    Set Model Training Parameters Below!
"""

"""Do what? [0 - both, 1 - YOLO ONLY, 2 - ResNet50v2 ONLY]"""
pchoice = 0


# Naming convention for Models
dataset_name = "Synthetic"
# Epoch numbers
Model_epoch = 20

# YOLO Yaml File
Yolo_Yaml = r"d:\Documents\ZZ_Datasets\New_synthetic\data.yaml"
# Freeze Resnet Layers?
freeze_layers = True

# Dataset paths for Resnet

# Dataset_home_dir = "C:/Users/dei/Documents/Programming/Datasets/Combined_Dataset_ResNet"

# FOR UBUNTU IN DEI COMP
Dataset_home_dir = "/mnt/d/Documents/Z_Cleaned_Dataset_v2"

def Yolo_train():
    Model_name = "YOLOv8s(" + dataset_name + ")_e" + str(Model_epoch) + "_"+ str(date.today())
    # Create a new YOLO model from scratch
    model = YOLO("yolov8s.pt")
    # model = YOLO("runs/detect/YOLOv8s(Synthetic)_e10_2025-04-193/weights/best.pt")

    # Display model information (optional)
    model.info()

    # # Train the model
    model = model.train(data=Yolo_Yaml, epochs=Model_epoch, device='0',
                        save_period= 1, name=Model_name, single_cls = True,
                        cache= 'disk', freeze =list(range(20)))
    
    # model = model.train(
    #     name="custom_10",
    #     data=Yolo_Yaml,
    #     epochs=Model_epoch,
    #     device='0',
    #     save_period= 1,
    #     cache='disk',
    #     single_cls = True,
    #     save= True,
    #     # freeze = 20,
    #     imgsz= 720,
    #     lr0= 0.01173,
    #     lrf= 0.00536,
    #     momentum= 0.91939,
    #     weight_decay= 0.00034,
    #     warmup_epochs= 3.23214,
    #     warmup_momentum= 0.75301,
    #     box= 7.06289,
    #     cls= 0.53039,
    #     dfl= 1.64606,
    #     hsv_h= 0.01452,
    #     hsv_s= 0.76116,
    #     hsv_v= 0.23249,
    #     degrees= 0.0,
    #     translate= 0.07547,
    #     scale= 0.36605,
    #     shear= 0.0,
    #     perspective= 0.0,
    #     flipud= 0.0,
    #     fliplr= 0.62907,
    #     bgr= 0.0,
    #     mosaic= 1,
    #     mixup= 0.0,
    #     copy_paste= 0.0
    # )

    # model.export(format="pt")
    
def Resnet_train():
    if not freeze_layers:
        Model_title = "Resnet50V2(newgen_" + str(date.today()) + ")_" + str(Model_epoch) + "e_uf20_adam.keras"
    else:
        Model_title = "Resnet50V2(newgen_" + str(date.today()) + ")_" + str(Model_epoch) + "e_adam.keras"

    train_dir =  Dataset_home_dir + "/train"  # Replace with your dataset path
    val_dir = Dataset_home_dir + "/test"      # Replace with your dataset path

    train_datagen = ImageDataGenerator(
        dtype = 'float32',
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
        rotation_range=15,       # Randomly rotate images by up to 30 degrees
        width_shift_range=0.2,   # Randomly shift width by 20%
        height_shift_range=0.2,  # Randomly shift height by 20%
        shear_range=0.2,         # Shearing transformations
        zoom_range=0.2,          # Random zoom
        horizontal_flip=True,    # Flip images horizontally
        fill_mode='nearest'      # Fill missing pixels
        )

    val_datagen = ImageDataGenerator(dtype = 'float32', preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)

    img_size = 224

    # Load datasets
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

    # Load ResNet50V2 with pretrained ImageNet weights, exclude the top layer
    base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

    # Unfreeze the last few layers of ResNetV2 for fine-tuning
    if not freeze_layers:
        for layer in base_model.layers[-20:]:  # Unfreezing last 20 layers
            layer.trainable = True
    else:
        # Freeze the base model
        for layer in base_model.layers:
            layer.trainable = False

    # Add custom layers for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(train_dataset.num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    print(model.summary())

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',   # Watch validation loss
        factor=0.5,           # Reduce LR by half if no improvement
        patience=3,           # Wait 3 epochs before reducing LR
        min_lr=1e-6,           # Set a minimum LR limit
        verbose = 1
    )

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy','recall','precision'])
    
    # Enabling memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=Model_epoch
    )

    model.save(Model_title)
    
def Yolo_cont():
    # insert path to last.pt to continue
    model = YOLO("runs/detect/YOLOv8s(Synthetic)_e70_2025-04-14/weights/last.pt")
    results =  model.train(resume= True, data = r"runs/detect/YOLOv8s(Synthetic)_e70_2025-04-14/args.yaml")
    # results = model.tune(resume=True, data= Yolo_Yaml)
    # results = model.tune(resume=True,patience = 5)
    # model.export(format="pt")
    
def Yolo_tune():
    
    model = YOLO("yolov8s.pt")
    
    model.tune(name="Tune(Edited_Dataset)_10e_10i",
                data=Yolo_Yaml,
                epochs= 20,
                iterations= 30,
                optimizer= 'SGD',
                imgsz= 640,
                plots=True,
                save=True,
                cache='disk',
                )
    
if __name__ == "__main__":
    
    choice = ""
    
    while True:
        if not choice == '/n':
            choice = input("""
                        Do what?
                        0 - Train Both
                        1 - Train a YOLO Model
                        2 - Train a ResNet50v2 Model
                        3 - Check GPU's
                        4 - Continue last YOLO Training
                        5 - Tune a YOLO Model 
                        """)
        
        if choice == '/n':
            choice = pchoice
        elif choice == '0':
            Yolo_train()
            Resnet_train()
            exit()
        elif choice == '1':
            Yolo_train()
            exit()
        elif choice == '2':
            Resnet_train()
            exit()
        elif choice == '3':
            print('WIP')
            pass
        elif choice == '4':
            Yolo_cont()
        elif choice == '5':
            Yolo_tune()
        else:
            print("invalid choice!")
