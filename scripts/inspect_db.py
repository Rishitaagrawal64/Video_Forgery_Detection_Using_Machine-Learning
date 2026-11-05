import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("FORGERY_DATABASE_URL", "postgresql+psycopg://forgery:forgery@localhost:5432/forgery")

def inspect_database():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        print("=== VIDEOS TABLE ===")
        videos = conn.execute(text("SELECT id, filename, status, created_at FROM videos;")).fetchall()
        for v in videos:
            print(dict(v._mapping))

        print("\n=== PREDICTIONS TABLE ===")
        preds = conn.execute(text("SELECT video_id, status, label, processing_sec FROM predictions;")).fetchall()
        for p in preds:
            print(dict(p._mapping))

if __name__ == "__main__":
    inspect_database()