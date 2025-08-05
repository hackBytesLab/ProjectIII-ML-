import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# --- Global Settings ---
dataset_path = 'dataset'
camera_index = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# CNN settings
img_size = (200, 200)  # Resize ทุกภาพให้เท่ากัน

# --- Capture Faces ---
def capture_faces():
    person_name = input("Enter the person's name (no spaces): ").strip()
    if not person_name:
        print("❌ Invalid name.")
        return

    save_path = os.path.join(dataset_path, person_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam at index {camera_index}.")
        return

    print("\nLook at the camera. The script will start capturing faces.")
    print("Press 'q' to quit early.")

    count = 0
    samples_to_take = 150
    while count < samples_to_take:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            (x, y, w, h) = faces[0]
            face_img = cv2.resize(gray[y:y+h, x:x+w], img_size)
            face_image_path = os.path.join(save_path, f"{count+1}.jpg")
            cv2.imwrite(face_image_path, face_img)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text = f"Saved: {count} / {samples_to_take}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("User quit capture early.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished capturing {count} images for '{person_name}'.")

# --- Train Model (CNN) ---
def train_model():
    print("\n🔄 Loading images and training CNN model...")
    X, y = [], []

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ Dataset folder is empty or does not exist. Please capture faces first.")
        return

    for folder_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(person_path):
            continue

        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, img_size)
            X.append(image)
            y.append(folder_name)

    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1) / 255.0

    if len(X) < 10:
        print("❌ Not enough training data. Please add more faces.")
        return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Build CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_categorical, epochs=20, batch_size=16, validation_split=0.2)

    model.save('cnn_model.h5')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("✅ Training complete! CNN model saved as 'cnn_model.h5'.")

# --- Recognize Faces ---
def recognize_faces():
    try:
        model = load_model('cnn_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("❌ Model not found. Please train the model first (Option 2).")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam at index {camera_index}.")
        return

    print("\nStarting real-time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], img_size) / 255.0
            face_roi = face_roi.reshape(1, img_size[0], img_size[1], 1)

            predictions = model.predict(face_roi)
            confidence = np.max(predictions) * 100
            label_idx = np.argmax(predictions)

            if confidence > 85:
                name = le.inverse_transform([label_idx])[0]
                text = f"{name} ({confidence:.1f}%)"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Face Recognition (CNN)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User quit recognition.")
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Delete Person Data ---
def delete_person_data():
    person_name = input("Enter the name of the person's data you want to delete: ").strip()
    if not person_name:
        print("❌ Invalid name.")
        return

    person_path = os.path.join(dataset_path, person_name)
    if os.path.exists(person_path):
        try:
            shutil.rmtree(person_path)
            print(f"✅ Successfully deleted dataset for '{person_name}'.")
            print("⚠️ Please retrain the model (Option 2) to update changes.")
        except OSError as e:
            print(f"❌ Error deleting folder {person_path}: {e.strerror}")
    else:
        print(f"❌ Dataset for '{person_name}' not found.")

# --- Main Menu ---
def main():
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    while True:
        print("\n--- Face Recognition Menu ---")
        print("1. Capture new faces")
        print("2. Train the CNN model")
        print("3. Recognize faces")
        print("4. Delete a person's data")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            capture_faces()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            delete_person_data()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
    
