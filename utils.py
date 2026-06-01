import cv2
import numpy as np
from keras_facenet import FaceNet

# Load FaceNet embedder once (CNN embeddings)
embedder = FaceNet()

# Simple face detector (offline)
_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(_cascade_path)

def detect_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces  # list of (x, y, w, h)

def extract_face(frame_bgr, box, required_size=(160, 160)):
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    face_bgr = frame_bgr[y:y+h, x:x+w]
    face_bgr = cv2.resize(face_bgr, required_size)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return face_rgb

def get_embedding(face_rgb):
    face = np.asarray(face_rgb, dtype=np.float32)
    face = np.expand_dims(face, axis=0)
    emb = embedder.embeddings(face)[0]  # 1D vector

    # Normalize (good for cosine similarity)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-10) * (np.linalg.norm(b) + 1e-10)))

def find_best_match(query_emb, db_items, threshold=0.65):
    """
    db_items: list of (student_id, code, name, embedding_list)
    returns: (student_id, code, name, score) or None
    """
    best = None
    best_score = -1.0

    for student_id, code, name, emb_list in db_items:
        score = cosine_similarity(query_emb, emb_list)
        if score > best_score:
            best_score = score
            best = (student_id, code, name, score)

    if best and best[3] >= threshold:
        return best
    return None