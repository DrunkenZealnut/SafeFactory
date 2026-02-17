"""
Metadata Manager for Pinecone Agent
Manages metadata storage in SQLite database for tracking uploaded files.
Uses the same instance/app.db as the web application.
"""

import atexit
import os
import hashlib
import json
import sqlite3
import threading
import weakref
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class MetadataManager:
    """Manages metadata storage for Pinecone uploaded files using SQLite.

    Thread-safe: each operation acquires its own connection via a
    per-thread local or a short-lived context manager, avoiding the
    'SQLite objects created in a thread' error.
    """

    def __init__(self, db_path: str = None):
        """Initialize database path and ensure schema exists.

        Args:
            db_path: Path to SQLite database file.
                     Defaults to instance/app.db relative to project root.
        """
        if db_path is None:
            basedir = Path(__file__).resolve().parent.parent
            db_path = str(basedir / 'instance' / 'app.db')

        self.db_path = db_path
        self._local = threading.local()
        self._connections: weakref.WeakSet = weakref.WeakSet()
        self._conn_lock = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.create_table_if_not_exists()
        atexit.register(self.close_all)
        print(f"\u2713 MetadataManager ready: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Yield a thread-local SQLite connection.

        Connections are cached per-thread and reused for the thread's
        lifetime, which is safe for SQLite.
        """
        conn = getattr(self._local, 'connection', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.connection = conn
            with self._conn_lock:
                self._connections.add(conn)
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    def create_table_if_not_exists(self):
        """Create pinecone_agent table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS pinecone_agent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL,
            source_file TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            vector_count INTEGER DEFAULT 0,
            vector_ids TEXT,
            upload_date TEXT,
            last_modified TEXT,
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
            error_message TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(namespace, source_file)
        );
        """

        create_index_sqls = [
            "CREATE INDEX IF NOT EXISTS idx_pa_namespace ON pinecone_agent(namespace);",
            "CREATE INDEX IF NOT EXISTS idx_pa_source_file ON pinecone_agent(source_file);",
            "CREATE INDEX IF NOT EXISTS idx_pa_file_hash ON pinecone_agent(file_hash);",
            "CREATE INDEX IF NOT EXISTS idx_pa_status ON pinecone_agent(status);",
        ]

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_sql)
                for idx_sql in create_index_sqls:
                    cursor.execute(idx_sql)
                conn.commit()
            print("\u2713 Table 'pinecone_agent' is ready")
        except Exception as e:
            print(f"\u2717 Failed to create table: {e}")

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Warning: Failed to calculate hash for {file_path}: {e}")
            return ""

    def _row_to_dict(self, row) -> Optional[Dict]:
        """Convert a sqlite3.Row to a plain dict."""
        if row is None:
            return None
        return dict(row)

    def get_file_metadata(self, namespace: str, source_file: str) -> Optional[Dict]:
        """Return metadata dict for a file, or None if not found."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                sql = "SELECT * FROM pinecone_agent WHERE namespace = ? AND source_file = ?"
                cursor.execute(sql, (namespace, source_file))
                return self._row_to_dict(cursor.fetchone())
        except Exception as e:
            print(f"Error checking file existence: {e}")
            return None

    def file_changed(self, namespace: str, source_file: str, current_hash: str) -> bool:
        """Check if file has changed since last upload."""
        existing = self.get_file_metadata(namespace, source_file)
        if not existing:
            return True  # New file
        return existing['file_hash'] != current_hash

    def insert_metadata(
        self,
        namespace: str,
        source_file: str,
        file_type: str,
        file_path: str,
        chunk_count: int = 0,
        vector_count: int = 0,
        vector_ids: List[str] = None,
        status: str = 'pending',
        error_message: str = None,
        file_hash: str = None,
        file_size: int = None
    ) -> bool:
        """Insert or update file metadata."""
        # Use provided values or calculate from file_path
        if file_hash is None:
            file_hash = self.calculate_file_hash(file_path)
        if file_size is None:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        last_modified = (
            datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            if os.path.exists(file_path) else None
        )
        upload_date = datetime.now().isoformat() if status == 'completed' else None
        vector_ids_json = json.dumps(vector_ids) if vector_ids else None
        now = datetime.now().isoformat()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                sql = """
                INSERT INTO pinecone_agent
                    (namespace, source_file, file_type, file_hash, file_size,
                     chunk_count, vector_count, vector_ids, upload_date,
                     last_modified, status, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, source_file) DO UPDATE SET
                    file_type = excluded.file_type,
                    file_hash = excluded.file_hash,
                    file_size = excluded.file_size,
                    chunk_count = excluded.chunk_count,
                    vector_count = excluded.vector_count,
                    vector_ids = excluded.vector_ids,
                    upload_date = excluded.upload_date,
                    last_modified = excluded.last_modified,
                    status = excluded.status,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """
                cursor.execute(sql, (
                    namespace, source_file, file_type, file_hash, file_size,
                    chunk_count, vector_count, vector_ids_json, upload_date,
                    last_modified, status, error_message, now, now
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting metadata: {e}")
            return False

    def get_all_metadata(self, namespace: str = None) -> List[Dict]:
        """Get all metadata records, optionally filtered by namespace."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if namespace:
                    sql = "SELECT * FROM pinecone_agent WHERE namespace = ? ORDER BY upload_date DESC"
                    cursor.execute(sql, (namespace,))
                else:
                    sql = "SELECT * FROM pinecone_agent ORDER BY upload_date DESC"
                    cursor.execute(sql)
                return [self._row_to_dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return []

    def get_stats(self, namespace: str = None) -> Dict:
        """Get statistics about stored metadata."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if namespace:
                    sql = """
                    SELECT
                        COUNT(*) as total_files,
                        COALESCE(SUM(chunk_count), 0) as total_chunks,
                        COALESCE(SUM(vector_count), 0) as total_vectors,
                        COALESCE(SUM(file_size), 0) as total_size,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                    FROM pinecone_agent WHERE namespace = ?
                    """
                    cursor.execute(sql, (namespace,))
                else:
                    sql = """
                    SELECT
                        COUNT(*) as total_files,
                        COALESCE(SUM(chunk_count), 0) as total_chunks,
                        COALESCE(SUM(vector_count), 0) as total_vectors,
                        COALESCE(SUM(file_size), 0) as total_size,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                    FROM pinecone_agent
                    """
                    cursor.execute(sql)
                result = cursor.fetchone()
                return self._row_to_dict(result) or {}
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

    def delete_metadata(self, namespace: str, source_file: str) -> bool:
        """Delete metadata for a specific file."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                sql = "DELETE FROM pinecone_agent WHERE namespace = ? AND source_file = ?"
                cursor.execute(sql, (namespace, source_file))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting metadata: {e}")
            return False

    def close(self):
        """Close the current thread's database connection."""
        conn = getattr(self._local, 'connection', None)
        if conn:
            with self._conn_lock:
                self._connections.discard(conn)
            conn.close()
            self._local.connection = None

    def close_all(self):
        """Close all tracked connections (called on process exit)."""
        with self._conn_lock:
            for conn in list(self._connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()


if __name__ == "__main__":
    """Test metadata manager."""
    from dotenv import load_dotenv
    load_dotenv()

    # Test connection
    manager = MetadataManager()

    # Test insert
    success = manager.insert_metadata(
        namespace="test",
        source_file="test/file.md",
        file_type="markdown",
        file_path=__file__,  # Use this file for testing
        chunk_count=5,
        vector_count=5,
        vector_ids=["vec1", "vec2", "vec3", "vec4", "vec5"],
        status="completed"
    )
    result = "\u2713 Success" if success else "\u2717 Failed"
    print(f"Insert test: {result}")

    # Test get stats
    stats = manager.get_stats("test")
    print(f"\nStats: {stats}")

    # Test get all
    records = manager.get_all_metadata("test")
    print(f"\nRecords: {len(records)}")

    manager.close()
