from tensorflow.keras.preprocessing.image import ImageDataGenerator

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    r'D:\Documents\ZZ_Datasets\Resnet_GTSRB_Cleaned_FINAL(4-20-25)\valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.models import load_model

model = load_model('Resnet50V2(NewSyn_2025-04-22)_1e.keras')
loss, accuracy, precision, recall = model.evaluate(val_generator)

# Calculate F1 score
f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)  # small epsilon to avoid division by zero

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
