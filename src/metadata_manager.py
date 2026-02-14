"""
Metadata Manager for Pinecone Agent
Manages metadata storage in MySQL database for tracking uploaded files.
"""

import os
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import pymysql
from pymysql.cursors import DictCursor


class MetadataManager:
    """Manages metadata storage for Pinecone uploaded files."""

    def __init__(self):
        """Initialize database connection."""
        self.connection = None
        self.connect()
        self.create_table_if_not_exists()

    def connect(self):
        """Connect to MySQL database."""
        try:
            self.connection = pymysql.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 3306)),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', ''),
                database=os.getenv('DB_NAME', 'kcsvictory'),
                charset='utf8mb4',
                cursorclass=DictCursor
            )
            print(f"✓ Connected to MySQL database: {os.getenv('DB_NAME', 'kcsvictory')}")
        except Exception as e:
            print(f"✗ Failed to connect to MySQL: {e}")
            self.connection = None

    def create_table_if_not_exists(self):
        """Create pinecone_agent table if it doesn't exist."""
        if not self.connection:
            return

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS pinecone_agent (
            id INT AUTO_INCREMENT PRIMARY KEY,
            namespace VARCHAR(255) NOT NULL,
            source_file VARCHAR(500) NOT NULL,
            file_type ENUM('image', 'markdown', 'json') NOT NULL,
            file_hash VARCHAR(64) NOT NULL,
            file_size BIGINT NOT NULL,
            chunk_count INT DEFAULT 0,
            vector_count INT DEFAULT 0,
            vector_ids TEXT,
            upload_date DATETIME,
            last_modified DATETIME,
            status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_namespace (namespace),
            INDEX idx_source_file (source_file),
            INDEX idx_file_hash (file_hash),
            INDEX idx_status (status),
            UNIQUE KEY unique_file (namespace, source_file)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_table_sql)
            self.connection.commit()
            print("✓ Table 'pinecone_agent' is ready")
        except Exception as e:
            print(f"✗ Failed to create table: {e}")

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

    def file_exists(self, namespace: str, source_file: str) -> Optional[Dict]:
        """Check if file metadata already exists."""
        if not self.connection:
            return None

        try:
            with self.connection.cursor() as cursor:
                sql = "SELECT * FROM pinecone_agent WHERE namespace = %s AND source_file = %s"
                cursor.execute(sql, (namespace, source_file))
                return cursor.fetchone()
        except Exception as e:
            print(f"Error checking file existence: {e}")
            return None

    def file_changed(self, namespace: str, source_file: str, current_hash: str) -> bool:
        """Check if file has changed since last upload."""
        existing = self.file_exists(namespace, source_file)
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
        if not self.connection:
            return False

        # Use provided values or calculate from file_path
        if file_hash is None:
            file_hash = self.calculate_file_hash(file_path)
        if file_size is None:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path)) if os.path.exists(file_path) else None
        upload_date = datetime.now() if status == 'completed' else None
        vector_ids_json = json.dumps(vector_ids) if vector_ids else None

        try:
            with self.connection.cursor() as cursor:
                # Check if record exists
                existing = self.file_exists(namespace, source_file)

                if existing:
                    # Update existing record
                    sql = """
                    UPDATE pinecone_agent
                    SET file_type = %s, file_hash = %s, file_size = %s,
                        chunk_count = %s, vector_count = %s, vector_ids = %s,
                        upload_date = %s, last_modified = %s, status = %s,
                        error_message = %s
                    WHERE namespace = %s AND source_file = %s
                    """
                    cursor.execute(sql, (
                        file_type, file_hash, file_size, chunk_count, vector_count,
                        vector_ids_json, upload_date, last_modified, status, error_message,
                        namespace, source_file
                    ))
                else:
                    # Insert new record
                    sql = """
                    INSERT INTO pinecone_agent
                    (namespace, source_file, file_type, file_hash, file_size,
                     chunk_count, vector_count, vector_ids, upload_date,
                     last_modified, status, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        namespace, source_file, file_type, file_hash, file_size,
                        chunk_count, vector_count, vector_ids_json, upload_date,
                        last_modified, status, error_message
                    ))

            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error inserting metadata: {e}")
            self.connection.rollback()
            return False

    def get_all_metadata(self, namespace: str = None) -> List[Dict]:
        """Get all metadata records, optionally filtered by namespace."""
        if not self.connection:
            return []

        try:
            with self.connection.cursor() as cursor:
                if namespace:
                    sql = "SELECT * FROM pinecone_agent WHERE namespace = %s ORDER BY upload_date DESC"
                    cursor.execute(sql, (namespace,))
                else:
                    sql = "SELECT * FROM pinecone_agent ORDER BY upload_date DESC"
                    cursor.execute(sql)
                return cursor.fetchall()
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return []

    def get_stats(self, namespace: str = None) -> Dict:
        """Get statistics about stored metadata."""
        if not self.connection:
            return {}

        try:
            with self.connection.cursor() as cursor:
                if namespace:
                    sql = """
                    SELECT
                        COUNT(*) as total_files,
                        SUM(chunk_count) as total_chunks,
                        SUM(vector_count) as total_vectors,
                        SUM(file_size) as total_size,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                    FROM pinecone_agent WHERE namespace = %s
                    """
                    cursor.execute(sql, (namespace,))
                else:
                    sql = """
                    SELECT
                        COUNT(*) as total_files,
                        SUM(chunk_count) as total_chunks,
                        SUM(vector_count) as total_vectors,
                        SUM(file_size) as total_size,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                    FROM pinecone_agent
                    """
                    cursor.execute(sql)
                return cursor.fetchone() or {}
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

    def delete_metadata(self, namespace: str, source_file: str) -> bool:
        """Delete metadata for a specific file."""
        if not self.connection:
            return False

        try:
            with self.connection.cursor() as cursor:
                sql = "DELETE FROM pinecone_agent WHERE namespace = %s AND source_file = %s"
                cursor.execute(sql, (namespace, source_file))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error deleting metadata: {e}")
            self.connection.rollback()
            return False

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("✓ Database connection closed")


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
    print(f"Insert test: {'✓ Success' if success else '✗ Failed'}")

    # Test get stats
    stats = manager.get_stats("test")
    print(f"\nStats: {stats}")

    # Test get all
    records = manager.get_all_metadata("test")
    print(f"\nRecords: {len(records)}")

    manager.close()
