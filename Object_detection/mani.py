import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Step 1: Load and preprocess the dataset
dataset_dir = 'Object_detection/data_sets'
image_paths = []
images = []
labels = []

# Load all image paths and labels
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_paths.append(os.path.join(root, file))
            labels.append(os.path.basename(root))  # Extract the final directory name as the label

# Load images
for path in image_paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    images.append(image)

# Convert images and labels to numpy arrays
X = np.array(images)
y = np.array(labels)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Preprocess the image data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 4: Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)
y_train_categorical = to_categorical(y_train_encoded, num_classes)
y_test_categorical = to_categorical(y_test_encoded, num_classes)

# Step 5: Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Step 6: Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_categorical, epochs=20, batch_size=32, validation_data=(X_test, y_test_categorical))

# Step 7: Capture and label live video frames
cap = cv2.VideoCapture(0)
# the window name and size
cv2.namedWindow('Live Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Video', 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = cv2.resize(frame, (100, 100))  # Resize frame to match training data size
    frame_data = np.expand_dims(frame, axis=0) / 255.0  # Expand dimensions and normalize

 # Predict label for the frame
    predicted_probs = model.predict(frame_data)[0]
    predicted_label = label_encoder.inverse_transform([np.argmax(predicted_probs)])[0]

    # Display the frame with predicted label
    cv2.putText(frame, predicted_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2 )
    

# increase the quality of output screen
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR_EXACT)
    cv2.imshow('Live Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
