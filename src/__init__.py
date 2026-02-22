# Pinecone Agent Package


class HttpClientMixin:
    """Mixin providing HTTP client lifecycle management and context manager support."""

    def close(self):
        """Close the underlying HTTP client."""
        if hasattr(self, '_http_client'):
            try:
                self._http_client.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
