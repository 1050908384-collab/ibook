import sys
sys.path.insert(0, 'backend')
from db.sqlite_client import SQLiteClient
from sqlalchemy import text

db = SQLiteClient()
conn = db.engine.connect()

tables = ['knowledge_blocks', 'knowledge_subtopics', 'knowledge_docs', 'faq_items', 'test_questions']
for table in tables:
    try:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"{table}: {count} rows")
    except Exception as e:
        print(f"{table}: ERROR - {e}")

conn.close()
db.close()
