import cv2
import os
import shutil
from deepface import DeepFace

# --- Global Settings ---
dataset_path = 'dataset'
camera_index = 0

# --- Capture Faces ---
def capture_faces():
    """Captures face samples from the webcam and saves them to the dataset."""
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
    samples_to_take = 200
    while count < samples_to_take:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_image_path = os.path.join(save_path, f"{count+1}.jpg")
            cv2.imwrite(face_image_path, face_img)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text = f"Saved: {count} / {samples_to_take}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Saved image {count} to {face_image_path}")

        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print("User quit capture early.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished capturing {count} images for '{person_name}'.")

# --- Recognize Faces ---
def recognize_faces():
    """Recognizes faces in real-time using DeepFace."""
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ Dataset is empty. Please capture faces first.")
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

        try:
            result = DeepFace.find(img_path=frame, db_path=dataset_path, model_name="Facenet", enforce_detection=False)

            if len(result[0]) > 0:
                identity = result[0].iloc[0]["identity"]
                person_name = os.path.basename(os.path.dirname(identity))
                text = person_name
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)
        except:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow('Face Recognition', frame)

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
        print("\n--- Face Recognition Menu ---")
        print("1. Capture new faces")
        print("2. Recognize faces")
        print("3. Delete a person's data")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            capture_faces()
        elif choice == '2':
            recognize_faces()
        elif choice == '3':
            delete_person_data()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
