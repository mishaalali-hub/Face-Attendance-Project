import cv2
import numpy as np
from db import upsert_student, save_embedding
from utils import detect_faces, extract_face, get_embedding

def enroll_one_student(samples_needed=20):
    student_code = input("\nEnter Student ID (e.g., TP12345): ").strip()
    full_name = input("Enter Student Name: ").strip()

    if not student_code or not full_name:
        print("[X] Student ID and Name cannot be empty.")
        return

    student_id = upsert_student(student_code, full_name)
    print(f"[OK] Student saved. DB student_id = {student_id}")

    cap = cv2.VideoCapture(0)
    embeddings = []

    print("\nLook at the camera.")
    print("Press 'c' to capture a face sample.")
    print("Press 'q' to cancel enrollment.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[X] Camera not working.")
            break

        faces = detect_faces(frame)

        # Draw boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"Captured: {len(embeddings)}/{samples_needed}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Enroll Student", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if len(faces) == 0:
                print("[!] No face detected. Try again.")
                continue

            # Use the biggest face
            faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            face_rgb = extract_face(frame, faces_sorted[0])
            emb = get_embedding(face_rgb)
            embeddings.append(emb)

            print(f"[OK] Captured sample {len(embeddings)}/{samples_needed}")

            if len(embeddings) >= samples_needed:
                break

        if key == ord('q'):
            print("[!] Enrollment cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < 5:
        print("[X] Not enough samples collected. Try again.")
        return

    mean_emb = np.mean(np.stack(embeddings), axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)

    save_embedding(student_id, mean_emb)
    print(f"[DONE] Enrollment complete for {full_name} ({student_code}).")

def main():
    while True:
        enroll_one_student(samples_needed=20)
        again = input("\nEnroll another student? (y/n): ").strip().lower()
        if again != "y":
            break

if __name__ == "__main__":
    main()