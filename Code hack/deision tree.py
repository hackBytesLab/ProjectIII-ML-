import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Global Settings ---
dataset_path = 'dataset'
camera_index = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# LBP settings for feature extraction
radius = 1
n_points = 8 * radius

def capture_faces():
    """Captures face samples from the webcam and saves them to the dataset."""
    person_name = input("Enter the person's name (no spaces): ")
    save_path = os.path.join(dataset_path, person_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}.")
        return

    print("\nLook at the camera. The script will start capturing faces.")
    print("Press 'q' to quit early.")
    
    count = 0
    samples_to_take = 300
    while count < samples_to_take:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            face_image_path = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(face_image_path, gray[y:y+h, x:x+w])
            
            text = f"Saved: {count} / {samples_to_take}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Saved image {count} to {face_image_path}")

        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished capturing {count} images for {person_name}.")


def train_model():
    """Trains the Decision Tree model on the collected dataset."""
    features = []
    labels = []
    print("Loading images and training model...")

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("Dataset folder is empty or does not exist. Please capture faces first.")
        return

    for folder_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(person_path): continue

        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: continue

            lbp = local_binary_pattern(image, n_points, radius, method='uniform')
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            
            features.append(hist)
            labels.append(folder_name)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    # 🔥 เปลี่ยนจาก SVM เป็น Decision Tree
    model = DecisionTreeClassifier(criterion='entropy', max_depth=20, random_state=42)
    model.fit(features, y_encoded)

    with open('decision_tree_model.pkl', 'wb') as f: pickle.dump(model, f)
    with open('label_encoder.pkl', 'wb') as f: pickle.dump(le, f)
    
    print("\nTraining complete! Model saved as 'decision_tree_model.pkl'.")


def recognize_faces():
    """Recognizes faces in real-time using the trained Decision Tree model."""
    try:
        with open('decision_tree_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f: le = pickle.load(f)
    except FileNotFoundError:
        print("Model not found. Please train the model first (Option 2).")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}.")
        return

    print("Starting real-time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            lbp = local_binary_pattern(face_roi, n_points, radius, method='uniform')
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            features = hist.reshape(1, -1)

            prediction = model.predict(features)
            proba = model.predict_proba(features)
            confidence = proba.max() * 100
            predicted_name = le.inverse_transform(prediction)[0]
            
            text = f"Unknown ({confidence:.2f}%)"
            if confidence > 75:
                text = f"{predicted_name} ({confidence:.2f}%)"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function to display the menu and handle user choice."""
    while True:
        print("\n--- Face Recognition Menu ---")
        print("1. Capture new faces")
        print("2. Train the model")
        print("3. Recognize faces")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            capture_faces()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


if __name__ == "__main__":
    main()


