import json
from datetime import date, datetime
import mysql.connector

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Samir@2006",   # <-- change this to match your MySQL user
    "database": "face_attendance",  # the app database
}

SCHEMA_FILE = "schema.sql.sql"  # path to the SQL file included in repo

def get_conn():
    """Return a connection using :data:`DB_CONFIG`.

    If the requested database does not exist, attempt to create it by running
    the schema file.  This makes the application self-initializing.
    """
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.errors.ProgrammingError as e:
        # 1049 = unknown database
        if e.errno == 1049:
            _initialize_database()
            return mysql.connector.connect(**DB_CONFIG)
        raise


def _initialize_database():
    """Create the database and run the schema file.

    Connects without specifying a database, creates the database if missing, and
    executes any SQL found in :data:`SCHEMA_FILE`.
    """
    # connect without database to create it
    tmp_conf = DB_CONFIG.copy()
    tmp_conf.pop("database", None)
    conn = mysql.connector.connect(**tmp_conf)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    conn.commit()
    cur.close()
    conn.close()

    # run schema file (assumes file is valid SQL and uses USE ... statement)
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        sql = f.read()
    conn = mysql.connector.connect(**tmp_conf, database=DB_CONFIG["database"])
    cur = conn.cursor()
    for statement in sql.split(";"):
        stmt = statement.strip()
        if stmt:
            cur.execute(stmt)
    conn.commit()
    cur.close()
    conn.close()

def upsert_student(student_code: str, full_name: str) -> int:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM students WHERE student_code=%s", (student_code,))
    row = cur.fetchone()

    if row:
        student_id = row[0]
        cur.execute("UPDATE students SET full_name=%s WHERE id=%s", (full_name, student_id))
    else:
        cur.execute(
            "INSERT INTO students(student_code, full_name) VALUES (%s, %s)",
            (student_code, full_name),
        )
        student_id = cur.lastrowid

    conn.commit()
    cur.close()
    conn.close()
    return student_id

def save_embedding(student_id: int, embedding):
    conn = get_conn()
    cur = conn.cursor()

    emb_json = json.dumps([float(x) for x in embedding])
    cur.execute(
        "INSERT INTO face_embeddings(student_id, embedding_json) VALUES (%s, %s)",
        (student_id, emb_json),
    )

    conn.commit()
    cur.close()
    conn.close()

def load_all_embeddings():
    """
    Returns list of:
      (student_id, student_code, full_name, embedding_list)
    Note: If you store multiple embeddings per student, all will be returned.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT s.id, s.student_code, s.full_name, fe.embedding_json
        FROM students s
        JOIN face_embeddings fe ON fe.student_id = s.id
    """)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    out = []
    for student_id, code, name, emb_json in rows:
        out.append((student_id, code, name, json.loads(emb_json)))
    return out

def mark_attendance(student_id: int) -> bool:
    """
    Marks attendance once per day (UNIQUE student_id + date).
    Returns True if marked now, False if already marked today.
    """
    today = date.today()
    now_time = datetime.now().time().replace(microsecond=0)

    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO attendance(student_id, date, time_in) VALUES (%s, %s, %s)",
            (student_id, today, now_time),
        )
        conn.commit()
        return True
    except mysql.connector.errors.IntegrityError:
        return False
    finally:
        cur.close()
        conn.close()