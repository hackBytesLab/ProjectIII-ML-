import cv2
import os
import shutil
from roboflow import Roboflow

# --- Roboflow Settings ---
rf = Roboflow(api_key="YOUR_API_KEY")   # <-- ใส่ API Key ของคุณ
project = rf.workspace().project("your-project-name")  # <-- ชื่อโปรเจกต์
model = project.version(1).model         # <-- เวอร์ชันของโมเดล

# --- Global Settings ---
dataset_path = 'dataset'
camera_index = 0

# --- Capture Faces ---
def capture_faces():
    """Captures face samples from the webcam and saves them locally."""
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
    samples_to_take = 100
    while count < samples_to_take:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from camera.")
            break

        cv2.imshow('Capturing Faces', frame)
        face_image_path = os.path.join(save_path, f"{count + 1}.jpg")
        cv2.imwrite(face_image_path, frame)
        count += 1

        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("User quit capture early.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Finished capturing {count} images for '{person_name}'.")
    print("⚠️ Please upload these images to Roboflow for training.")

# --- Train Model (Guide) ---
def train_model():
    """Guide user to train the model in Roboflow."""
    print("\nRoboflow trains models in the cloud.")
    print("1. Go to https://app.roboflow.com/")
    print("2. Create a project or open your project.")
    print("3. Upload the images from the 'dataset' folder.")
    print("4. Annotate faces and train your model.")
    print("5. Update your model version in the script when training is done.")
    print("✅ After training, you can run 'Recognize Faces' option.")

# --- Recognize Faces ---
def recognize_faces():
    """Recognizes faces using Roboflow model in real-time."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam at index {camera_index}.")
        return

    print("\nStarting Roboflow-based face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict using Roboflow model
        result = model.predict(frame_rgb, confidence=40, overlap=30).json()

        if len(result["predictions"]) == 0:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        for pred in result["predictions"]:
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            label = pred["class"]
            confidence = pred["confidence"] * 100

            color = (0, 255, 0) if confidence > 90 else (0, 0, 255)
            text = f"{label} ({confidence:.1f}%)" if confidence > 90 else "Unknown"

            cv2.rectangle(frame, (x - w // 2, y - h // 2),
                          (x + w // 2, y + h // 2), color, 2)
            cv2.putText(frame, text, (x - w // 2, y - h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face Recognition (Roboflow)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User quit recognition.")
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Delete Person Data ---
def delete_person_data():
    """Deletes a person's entire dataset folder."""
    person_name = input("Enter the name of the person's data you want to delete: ").strip()
    if not person_name:
        print("❌ Invalid name.")
        return

    person_path = os.path.join(dataset_path, person_name)
    if os.path.exists(person_path):
        try:
            shutil.rmtree(person_path)
            print(f"✅ Successfully deleted dataset for '{person_name}'.")
        except OSError as e:
            print(f"❌ Error deleting folder {person_path}: {e.strerror}")
    else:
        print(f"❌ Dataset for '{person_name}' not found.")

# --- Main Menu ---
def main():
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    while True:
        print("\n--- Face Recognition Menu (Roboflow) ---")
        print("1. Capture new faces")
        print("2. Train the model (Guide)")
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
