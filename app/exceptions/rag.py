from typing import Optional, Dict, Any
from app.exceptions.base import RagException

class RagQueryFailedError(RagException):
    code = "RAG_QUERY_FAILED"
    status_code = 500

class RagEnrichmentFailedError(RagException):
    code = "RAG_ENRICHMENT_FAILED"
    status_code = 500
