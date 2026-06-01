# Face Recognition Attendance System (CNN Embeddings + OpenCV + MySQL)

A real-time attendance tracking system that identifies students using **CNN face embeddings (FaceNet)** and logs attendance automatically in **MySQL**. The system supports **student enrollment** (Student ID + Name) and **live recognition** via webcam with **duplicate prevention per day**.

---

## Features

* **Student Enrollment**

  * Prompts for **Student ID** and **Full Name**
  * Captures multiple face samples via webcam
  * Generates a **FaceNet embedding** and stores it in MySQL
* **Real-Time Recognition (Attendance Mode)**

  * Detects faces from live webcam feed using OpenCV
  * Extracts FaceNet embeddings and matches with stored embeddings (cosine similarity)
  * Displays **Name + Student ID** on screen
  * Marks attendance in MySQL with **one entry per student per day**
* **Database-backed**

  * Students table (ID, name)
  * Embeddings table (stored vectors)
  * Attendance table (date + time-in)

---

## Tech Stack

* **Python**
* **TensorFlow / Keras**
* **FaceNet embeddings** (`keras-facenet`)
* **OpenCV**
* **MySQL**
* **NumPy**

---

## Project Structure

```
face_attendance/
  db.py
  utils.py
  enroll.py
  recognize.py
  schema.sql
  requirements.txt
  README.md
```

---

## Installation

### 1) Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd face_attendance
```

### 2) (Recommended) Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then run activation again.

**Windows (CMD)**

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Database Setup (MySQL)

### 1) Create DB + tables

Run the script `schema.sql` using any of the following:

**MySQL Workbench**

* File → Open SQL Script → select `schema.sql`
* Click ⚡ Execute

**Terminal**

```bash
mysql -u root -p < schema.sql
```

### 2) Configure DB credentials

Open `db.py` and update:

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "YOUR_PASSWORD",
    "database": "face_attendance",
}
```

---

## Usage

### A) Enroll a new student (register face)

Run:

```bash
python enroll.py
```

* Enter **Student ID** and **Student Name**
* Webcam opens
* Press **c** to capture face samples
* Press **q** to cancel enrollment

### B) Start attendance (recognition mode)

Run:

```bash
python recognize.py
```

* Webcam opens
* If a student is recognized, the system displays:

  * `Name (StudentID) score MARKED/ALREADY`
* Press **q** to quit

---

## How Recognition Works (High Level)

1. OpenCV detects a face in the webcam frame
2. Face is cropped and resized (160×160)
3. FaceNet generates a **normalized embedding vector**
4. The embedding is compared against the database using **cosine similarity**
5. If similarity ≥ threshold (default `0.65`), the student is identified and attendance is recorded

---

## Configuration Notes

* Recognition threshold is in `recognize.py`:

  * `threshold = 0.65`
* If you get false positives/negatives, try:

  * `0.60` (more lenient)
  * `0.70` (more strict)

---

## Troubleshooting

### Camera not opening

* Close apps using the camera (Zoom/Teams/Camera)
* Try changing camera index:

  ```python
  cap = cv2.VideoCapture(1)
  ```

### “No embeddings found”

* You must run `python enroll.py` at least once to store embeddings.

### PowerShell activation blocked

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Future Improvements (Optional Enhancements)

* Upgrade face detection to **OpenCV DNN / MTCNN** for higher accuracy
* Add **liveness detection (anti-spoofing)**
* Build an **admin dashboard** (Flask/FastAPI) to view/export attendance
* Store embeddings as **BLOB** and move DB credentials into a `.env` file

