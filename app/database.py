import psycopg2
import psycopg2.extras
from app.config import settings

_conn = None


def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(settings.db_dsn)
        _conn.autocommit = True
    return _conn


def query(sql: str, params=None) -> list[dict]:
    conn = get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def execute(sql: str, params=None):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, params)


def execute_values(sql: str, rows: list, template: str = None, page_size: int = 100):
    conn = get_conn()
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, template=template, page_size=page_size)
