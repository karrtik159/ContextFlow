"""
Corpus lifecycle — the single write path into the retrieval corpus.

Every mutation of the corpus (ingest, replace, delete) goes through this
package, and every mutation ends by calling `corpus_changed`, which bumps the
tenant's corpus epoch and invalidates the caches derived from the corpus
(docs/ADVANCED_RAG_PLAN.md, invariant 10). A corpus write that bypasses this
package is a bug: it would leave the semantic cache serving answers grounded in
content that no longer exists.
"""

from app.services.corpus.events import (  # noqa: F401
    corpus_changed,
    get_corpus_epoch,
    get_corpus_epoch_safe,
)
from app.services.corpus.store import (  # noqa: F401
    IngestResult,
    compute_checksum,
    delete_document,
    ingest_document,
    list_documents,
    replace_document,
)
