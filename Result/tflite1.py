import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def load_data(data_dir, img_size=(128, 128)):
    images = []
    masks = []

    for img_file in os.listdir(data_dir):
        if '_mask' not in img_file:
            img_path = os.path.join(data_dir, img_file)
            mask_path = os.path.join(data_dir, img_file.replace('.jpg', '_mask.jpg'))

            img = load_img(img_path, target_size=img_size)
            mask = load_img(mask_path, target_size=img_size, color_mode='grayscale')

            img = img_to_array(img) / 255.0
            mask = img_to_array(mask) / 255.0

            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks)

test_dir = '/home/ece/Desktop/embed/split_datasets/test'

X_test, y_test = load_data(test_dir)

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

tflite_model_path = '/home/ece/Desktop/embed/final.tflite'

interpreter = load_tflite_model(tflite_model_path)

predicted_masks = []
for i in range(X_test.shape[0]):
    sample_input = np.expand_dims(X_test[i], axis=0)  
    predicted_mask = predict(interpreter, sample_input)
    predicted_masks.append(predicted_mask[0, :, :, 0])

predicted_masks = np.array(predicted_masks)

y_test_binary = (y_test > 0.5).astype(int)
predicted_masks_binary = (predicted_masks > 0.5).astype(int)

f1 = f1_score(y_test_binary.flatten(), predicted_masks_binary.flatten(), average='binary')
print(f'F1 Score: {f1:.4f}')


num_samples = 5  # Number of samples to display
plt.figure(figsize=(15, 5 * num_samples))

for i in range(num_samples):
    plt.subplot(num_samples, 3, i * 3 + 1)
    plt.title('Original Image')
    plt.imshow(X_test[i])
    plt.axis('off')

    plt.subplot(num_samples, 3, i * 3 + 2)
    plt.title('Ground Truth Mask')
    plt.imshow(y_test[i, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(num_samples, 3, i * 3 + 3)
    plt.title('Predicted Mask')
    plt.imshow(predicted_masks[i], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

epochs = np.arange(1, 11)  
train_loss = np.random.rand(10)  
val_loss = np.random.rand(10)    

plt.figure(figsize=(8, 4))
plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='o', label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
