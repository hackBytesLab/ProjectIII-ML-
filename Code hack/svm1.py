import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import shutil

# --- Global Settings ---
dataset_path = 'dataset'
camera_index = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# LBP settings
radius = 1
n_points = 8 * radius
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

# --- Train Model ---
def train_model():
    features = []
    labels = []
    print("\n🔄 Loading images and training model...")

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ Dataset folder is empty or does not exist. Please capture faces first.")
        return

    # Load dataset
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
            lbp = local_binary_pattern(image, n_points, radius, method='uniform')
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)

            features.append(hist)
            labels.append(folder_name)

    if len(features) < 10:
        print("❌ Not enough training data. Please add more faces.")
        return

    features = np.array(features)
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    # --- ตั้งค่าพารามิเตอร์ SVM เองตรงนี้ ---
    C_value = 5       # <-- เปลี่ยนค่า C ที่นี่ได้เลย
    gamma_value = 'scale'  # หรือจะเปลี่ยนเป็นตัวเลข เช่น 0.01 ก็ได้
    kernel_value = 'rbf'

    print(f"\nTraining SVM with C={C_value}, gamma={gamma_value}, kernel={kernel_value} ...")
    model = SVC(C=C_value, gamma=gamma_value, kernel=kernel_value, probability=True)
    model.fit(features, y_encoded)

    # (ถ้าต้องการเพิ่ม learning curve ให้เปิด comment ข้างล่าง)
    """
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, features, y_encoded, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label="Training Accuracy", marker='o')
    plt.plot(train_sizes, val_mean, label="Validation Accuracy", marker='s')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve for Face Recognition (SVM)")
    plt.legend()
    plt.grid(True)
    plt.show()
    """

    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("✅ Training complete! Model saved as 'svm_model.pkl'.")

# --- Recognize Faces ---
def recognize_faces():
    try:
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("❌ Model not found. Please train the model first (Option 2).")
        return
    except Exception as e:
        print(f"❌ Error loading model or label encoder: {e}")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam at index {camera_index}.")
        return

    print("\nStarting real-time recognition. Press 'q' to quit.")

    face_history = []          # History for face smoothing
    prediction_history = []    # History for prediction smoothing
    history_length = 5         # Number of frames to stabilize

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            face_history.append(faces[0])
            if len(face_history) > history_length:
                face_history.pop(0)

            # Smooth face box
            avg_x = int(np.mean([f[0] for f in face_history]))
            avg_y = int(np.mean([f[1] for f in face_history]))
            avg_w = int(np.mean([f[2] for f in face_history]))
            avg_h = int(np.mean([f[3] for f in face_history]))
            (x, y, w, h) = (avg_x, avg_y, avg_w, avg_h)

            face_roi = cv2.resize(gray[y:y+h, x:x+w], img_size)
            lbp = local_binary_pattern(face_roi, n_points, radius, method='uniform')
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            features = hist.reshape(1, -1)

            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)
            confidence = proba.max() * 100

            prediction_history.append((prediction, confidence))
            if len(prediction_history) > history_length:
                prediction_history.pop(0)

            names = [p[0] for p in prediction_history]
            final_prediction = max(set(names), key=names.count)
            avg_confidence = np.mean([p[1] for p in prediction_history if p[0] == final_prediction])

            color = (0, 0, 255)
            text = "Unknown"
            if avg_confidence > 85:
                predicted_name = le.inverse_transform([final_prediction])[0]
                text = f"{predicted_name} ({avg_confidence:.1f}%)"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            face_history.clear()
            prediction_history.clear()
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('Face Recognition', frame)
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
        print("2. Train the model")
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
