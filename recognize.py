import cv2
from db import load_all_embeddings, mark_attendance
from utils import detect_faces, extract_face, get_embedding, find_best_match

def main():
    db_items = load_all_embeddings()
    print(f"[INFO] Loaded {len(db_items)} embeddings from DB.")

    if len(db_items) == 0:
        print("[X] No embeddings found. Run enroll.py first.")
        return

    cap = cv2.VideoCapture(0)
    threshold = 0.65  # adjust if needed (0.60–0.75 typical)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[X] Camera not working.")
            break

        faces = detect_faces(frame)

        for box in faces:
            x, y, w, h = box
            face_rgb = extract_face(frame, box)
            emb = get_embedding(face_rgb)

            match = find_best_match(emb, db_items, threshold=threshold)

            if match:
                student_id, code, name, score = match
                marked = mark_attendance(student_id)
                status = "MARKED" if marked else "ALREADY"
                label = f"{name} ({code}) {score:.2f} {status}"
            else:
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, max(20, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Attendance - Press q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()