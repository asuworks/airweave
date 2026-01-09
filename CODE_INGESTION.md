# Code Ingestion for AI Agents

This document provides a comprehensive guide to understanding and implementing a robust code ingestion pipeline. It is designed to be read and understood by an AI agent, providing all the necessary code and explanations to replicate this process in another project.

The ingestion process is broken down into two main stages:

1.  **Chunking**: Breaking down large pieces of code into smaller, meaningful chunks.
2.  **Embedding**: Converting these chunks into numerical representations (vectors) that can be understood by machine learning models.

Finally, we will discuss how these embedded chunks are used for **Retrieval**.

## Chunking

Chunking is the process of breaking down large documents into smaller pieces. This is essential for language models, which have a limited context window. The goal is to create chunks that are both semantically coherent and small enough to be processed by the model.

Our approach uses a two-stage process:

1.  **High-level Chunking**: We first use a high-level chunking strategy to split the text into meaningful segments. For code, we use an AST-based chunker (`CodeChunker`), and for natural language, we use a semantic chunker (`SemanticChunker`).
2.  **Token-based Fallback**: If any of the chunks produced by the high-level chunker are still too large, we use a simple token-based chunker as a fallback to split them into smaller pieces. This ensures that no chunk exceeds the maximum token limit.

### Base Chunker

All chunkers implement the `BaseChunker` interface, which defines a single method: `chunk_batch`.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseChunker(ABC):
    """Interface for all chunker implementations."""

    @abstractmethod
    async def chunk_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Chunk a batch of texts asynchronously.

        Args:
            texts: List of textual representations to chunk

        Returns:
            List of chunk lists, where each chunk dict contains:
            {
                "text": str,           # Chunk text content
                "start_index": int,    # Start position in original text
                "end_index": int,      # End position in original text
                "token_count": int     # Number of tokens (cl100k_base tokenizer)
            }
        """
        pass
```

### Code Chunker

The `CodeChunker` uses Abstract Syntax Trees (ASTs) to split code into logical units like functions, classes, and methods. This is a more intelligent way to chunk code than simply splitting by lines or tokens, as it preserves the structure of the code.

```python
"""Code chunker using AST-based parsing with TokenChunker safety net."""

from typing import Any, Dict, List, Optional

from airweave.core.logging import logger
from airweave.platform.chunkers._base import BaseChunker, TiktokenWrapperForChonkie
from airweave.platform.sync.async_helpers import run_in_thread_pool
from airweave.platform.sync.exceptions import SyncFailureError


class CodeChunker(BaseChunker):
    """Singleton code chunker with AST-based parsing (no API calls).

    Two-stage approach (internal implementation detail):
    1. CodeChunker: Chunks at logical code boundaries (functions, classes, methods)
    2. TokenChunker fallback: Force-splits any oversized chunks at token boundaries

    The chunker is shared across all syncs in the pod to avoid reloading
    the Magika language detection model for every sync job.

    Note: Even with AST-based splitting, single large AST nodes (massive functions
    without children) can exceed chunk_size, so we use TokenChunker as safety net.
    """

    # Configuration constants
    MAX_TOKENS_PER_CHUNK = 8192  # OpenAI hard limit (safety net)
    CHUNK_SIZE = 2048  # Target chunk size (can be exceeded by large AST nodes)
    TOKENIZER = "cl100k_base"  # For accurate OpenAI token counting

    # Singleton instance
    _instance: Optional["CodeChunker"] = None

    def __new__(cls):
        """Singleton pattern - one instance per pod."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize once per pod (models load lazily on first use)."""
        if self._initialized:
            return

        self._code_chunker = None  # Lazy init
        self._token_chunker = None  # Lazy init (emergency fallback)
        self._tiktoken_tokenizer = None  # Lazy init
        self._initialized = True

        logger.debug(
            f"CodeChunker singleton initialized "
            f"(target: {self.CHUNK_SIZE}, hard_limit: {self.MAX_TOKENS_PER_CHUNK})"
        )

    def _ensure_chunkers(self):
        """Lazy initialization of chunker models.

        Initializes CodeChunker (AST parsing) + TokenChunker (safety net).

        Raises:
            SyncFailureError: If model loading fails (infrastructure error)
        """
        if self._code_chunker is not None:
            return

        try:
            import tiktoken
            from chonkie import CodeChunker as ChonkieCodeChunker
            from chonkie import TokenChunker

            # Initialize tiktoken tokenizer for accurate OpenAI token counting
            self._tiktoken_tokenizer = tiktoken.get_encoding(self.TOKENIZER)

            # Wrap tiktoken for Chonkie to handle special tokens like <|endoftext|>
            # Chonkie's internal AutoTokenizer doesn't pass allowed_special="all",
            # which causes failures when syncing code with special tokens
            tiktoken_wrapper = TiktokenWrapperForChonkie(self._tiktoken_tokenizer)

            # Initialize Chonkie's CodeChunker with auto language detection
            # Uses Magika (Google's ML-based language detector) to identify language
            self._code_chunker = ChonkieCodeChunker(
                language="auto",  # Auto-detect using Magika
                tokenizer=tiktoken_wrapper,  # Use wrapper that handles special tokens
                chunk_size=self.CHUNK_SIZE,
                include_nodes=False,
            )

            # Initialize TokenChunker for fallback
            # Splits at exact token boundaries when code chunking produces oversized chunks
            # GUARANTEES chunks ≤ MAX_TOKENS_PER_CHUNK (uses same tokenizer for encode/decode)
            self._token_chunker = TokenChunker(
                tokenizer=tiktoken_wrapper,  # Use wrapper that handles special tokens
                chunk_size=self.MAX_TOKENS_PER_CHUNK,
                chunk_overlap=0,
            )

            logger.info(
                f"Loaded CodeChunker (auto-detect, target: {self.CHUNK_SIZE}) + "
                f"TokenChunker fallback (hard_limit: {self.MAX_TOKENS_PER_CHUNK})"
            )

        except Exception as e:
            raise SyncFailureError(f"Failed to initialize CodeChunker: {e}")

    async def chunk_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Chunk a batch of code texts with two-stage approach.

        Stage 1: CodeChunker chunks at AST boundaries (functions, classes)
        Stage 1.5: Recount tokens with tiktoken cl100k_base (Chonkie reports incorrect counts)
        Stage 2: TokenChunker force-splits any chunks exceeding MAX_TOKENS_PER_CHUNK (hard limit)

        Uses run_in_thread_pool because Chonkie is synchronous (avoids blocking event loop).

        Args:
            texts: List of code textual representations to chunk

        Returns:
            List of chunk lists (one per input text), where each chunk is a dict

        Raises:
            SyncFailureError: If model initialization or batch processing fails
        """
        self._ensure_chunkers()

        # Stage 1: AST-based code chunking
        try:
            code_results = await run_in_thread_pool(self._code_chunker.chunk_batch, texts)
        except Exception as e:
            # CodeChunker failure = sync failure (not entity-level)
            raise SyncFailureError(f"CodeChunker batch processing failed: {e}")

        # Stage 1.5: Recount tokens with tiktoken (Chonkie's CodeChunker reports incorrect counts)
        # Chonkie counts tokens from individual AST nodes, but the final chunk text includes
        # whitespace/gaps between nodes plus leading/trailing content, causing underestimates.
        code_results_with_tiktoken = await run_in_thread_pool(
            self._recount_tokens_with_tiktoken, code_results
        )

        # Stage 2: Safety net (batched for efficiency, now uses accurate tiktoken counts)
        final_results = await run_in_thread_pool(
            self._apply_safety_net_batched, code_results_with_tiktoken
        )

        # Validate all chunks meet requirements
        for doc_chunks in final_results:
            for chunk in doc_chunks:
                # Check for empty chunks
                if not chunk["text"] or not chunk["text"].strip():
                    raise SyncFailureError("PROGRAMMING ERROR: Empty chunk produced by CodeChunker")

                # Check token limit enforced
                if chunk["token_count"] > self.MAX_TOKENS_PER_CHUNK:
                    raise SyncFailureError(
                        f"PROGRAMMING ERROR: Chunk has {chunk['token_count']} tokens "
                        f"after safety net (max: {self.MAX_TOKENS_PER_CHUNK})"
                    )

        return final_results

    def _apply_safety_net_batched(
        self, code_results: List[List[Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Split oversized chunks using TokenChunker fallback.

        Same implementation as SemanticChunker - collects oversized chunks,
        batch processes them, then reconstructs results.

        Args:
            code_results: Chunks from CodeChunker

        Returns:
            Final chunks as dicts, all guaranteed ≤ MAX_TOKENS_PER_CHUNK
        """
        # Collect oversized chunks with position mapping
        oversized_texts = []
        oversized_map = {}  # position in oversized_texts → (doc_idx, chunk_idx)

        for doc_idx, chunks in enumerate(code_results):
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.token_count > self.MAX_TOKENS_PER_CHUNK:
                    pos = len(oversized_texts)
                    oversized_texts.append(chunk.text)
                    oversized_map[pos] = (doc_idx, chunk_idx)

        # Batch process all oversized chunks with TokenChunker fallback
        # TokenChunker enforces hard limit in one pass (no recursion needed)
        split_results_by_position = {}
        if oversized_texts:
            logger.debug(
                f"Safety net: splitting {len(oversized_texts)} oversized code chunks "
                f"exceeding {self.MAX_TOKENS_PER_CHUNK} tokens with TokenChunker"
            )

            # Use TokenChunker to split at exact token boundaries
            # GUARANTEED to produce chunks ≤ MAX_TOKENS_PER_CHUNK in one pass
            split_results = self._token_chunker.chunk_batch(oversized_texts)
            split_results_by_position = dict(enumerate(split_results))

        # Reconstruct final results
        final_results = []
        for doc_idx, chunks in enumerate(code_results):
            final_chunks = []
            for chunk_idx, chunk in enumerate(chunks):
                # Check if this chunk was oversized
                oversized_pos = next(
                    (
                        pos
                        for pos, (d_idx, c_idx) in oversized_map.items()
                        if d_idx == doc_idx and c_idx == chunk_idx
                    ),
                    None,
                )

                if oversized_pos is not None:
                    # Replace with split sub-chunks
                    split_chunks = split_results_by_position[oversized_pos]
                    for sub_chunk in split_chunks:
                        final_chunks.append(self._convert_chunk(sub_chunk))
                else:
                    # Keep original chunk
                    final_chunks.append(self._convert_chunk(chunk))

            final_results.append(final_chunks)

        if oversized_texts:
            logger.debug(
                f"TokenChunker fallback split {len(oversized_texts)} code chunks "
                f"that exceeded {self.MAX_TOKENS_PER_CHUNK} tokens"
            )

        return final_results

    def _recount_tokens_with_tiktoken(self, code_results: List[List[Any]]) -> List[List[Any]]:
        """Recount all chunks with tiktoken cl100k_base for accurate token counts.

        Chonkie's CodeChunker reports incorrect token counts because it counts tokens
        from individual AST nodes, but the final chunk text includes:
        - Whitespace/gaps between AST nodes
        - Leading content before the first node
        - Trailing content after the last node

        This causes token counts to be significantly understated. We recount with
        tiktoken to get accurate token counts before the safety net check.

        Args:
            code_results: Chunks from CodeChunker with potentially incorrect token counts

        Returns:
            Same chunks but with token_count field updated to accurate tiktoken counts
        """
        for chunks in code_results:
            for chunk in chunks:
                # Recount with tiktoken (actual chunk text may be larger than reported)
                # Use allowed_special="all" to handle special tokens like <|endoftext|>
                # that may appear in code comments or strings
                chunk.token_count = len(
                    self._tiktoken_tokenizer.encode(chunk.text, allowed_special="all")
                )

        return code_results

    def _convert_chunk(self, chunk) -> Dict[str, Any]:
        """Convert Chonkie Chunk object to dict format.

        Token counts have already been recounted with tiktoken in
        _recount_tokens_with_tiktoken(), so we just use them directly.

        Args:
            chunk: Chonkie Chunk object with tiktoken token_count

        Returns:
            Dict with chunk data and accurate OpenAI token count
        """
        return {
            "text": chunk.text,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,  # Already tiktoken count
        }
```

### Semantic Chunker

The `SemanticChunker` is used for natural language text. It uses a local embedding model to find semantic boundaries in the text, and then splits the text at those boundaries. This results in chunks that are more coherent than what you would get from simply splitting by sentences or paragraphs.

```python
"""Semantic chunker using NeuralChunker with SentenceChunker safety net."""

from typing import Any, Dict, List, Optional

from airweave.core.logging import logger
from airweave.platform.chunkers._base import BaseChunker, TiktokenWrapperForChonkie
from airweave.platform.sync.async_helpers import run_in_thread_pool
from airweave.platform.sync.exceptions import SyncFailureError


class SemanticChunker(BaseChunker):
    """Singleton semantic chunker with local inference (no API calls).

    Two-stage chunking approach (internal implementation detail):
    1. SemanticChunker: Detects semantic boundaries via embedding similarity
    2. TokenChunker fallback: Force-splits any oversized chunks at token boundaries

    The chunker is shared across all syncs in the pod to avoid reloading
    the embedding model for every sync job.
    """

    # Configuration constants
    MAX_TOKENS_PER_CHUNK = 8192  # OpenAI text-embedding-3-small hard limit (safety net)
    SEMANTIC_CHUNK_SIZE = 4096  # Soft target for semantic groups (better search quality)
    OVERLAP_TOKENS = 128  # Token overlap between chunks

    # SemanticChunker configuration
    EMBEDDING_MODEL = "minishlab/potion-base-128M"  # Default: Good speed/quality balance

    SIMILARITY_THRESHOLD = 0.01  # 0-1: Lower=larger chunks, Higher=smaller chunks
    SIMILARITY_WINDOW = 10  # Number of sentences to compare for similarity
    MIN_SENTENCES_PER_CHUNK = 1  # Prevent tiny fragment chunks
    MIN_CHARACTERS_PER_SENTENCE = 24  # Minimum chars per sentence

    # Tokenizer for accurate OpenAI token counting
    TOKENIZER = "cl100k_base"

    # Singleton instance
    _instance: Optional["SemanticChunker"] = None

    def __new__(cls):
        """Singleton pattern - one instance per pod."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize once per pod (models load lazily on first use)."""
        if self._initialized:
            return

        self._semantic_chunker = None  # Lazy init
        self._token_chunker = None  # Lazy init (emergency fallback)
        self._tiktoken_tokenizer = None  # Lazy init
        self._initialized = True

        logger.debug(
            f"SemanticChunker singleton initialized "
            f"(model: {self.EMBEDDING_MODEL}, max_tokens: {self.MAX_TOKENS_PER_CHUNK})"
        )

    def _ensure_chunkers(self):
        """Lazy initialization of chunker models."""
        if self._semantic_chunker is not None:
            return

        try:
            import tiktoken
            from chonkie import SemanticChunker as ChonkieSemanticChunker
            from chonkie import TokenChunker

            self._tiktoken_tokenizer = tiktoken.get_encoding(self.TOKENIZER)
            tiktoken_wrapper = TiktokenWrapperForChonkie(self._tiktoken_tokenizer)

            self._semantic_chunker = ChonkieSemanticChunker(
                embedding_model=self.EMBEDDING_MODEL,
                chunk_size=self.SEMANTIC_CHUNK_SIZE,
                threshold=self.SIMILARITY_THRESHOLD,
                similarity_window=self.SIMILARITY_WINDOW,
                min_sentences_per_chunk=self.MIN_SENTENCES_PER_CHUNK,
                min_characters_per_sentence=self.MIN_CHARACTERS_PER_SENTENCE,
            )

            self._token_chunker = TokenChunker(
                tokenizer=tiktoken_wrapper,
                chunk_size=self.MAX_TOKENS_PER_CHUNK,
                chunk_overlap=0,
            )

            logger.info(
                f"Loaded SemanticChunker (model: {self.EMBEDDING_MODEL}, "
                f"target_size: {self.SEMANTIC_CHUNK_SIZE}, "
                f"threshold: {self.SIMILARITY_THRESHOLD}, "
                f"hard_limit: {self.MAX_TOKENS_PER_CHUNK}) + TokenChunker fallback"
            )

        except Exception as e:
            raise SyncFailureError(f"Failed to initialize chunkers: {e}")

    async def chunk_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]]:
        """Chunk a batch of texts with semantic chunking + TokenChunker fallback."""
        self._ensure_chunkers()

        try:
            semantic_results = await run_in_thread_pool(self._semantic_chunker.chunk_batch, texts)
        except Exception as e:
            raise SyncFailureError(f"SemanticChunker batch processing failed: {e}")

        semantic_results_with_tiktoken = await run_in_thread_pool(
            self._recount_tokens_with_tiktoken, semantic_results
        )

        final_results = await run_in_thread_pool(
            self._apply_safety_net_batched, semantic_results_with_tiktoken
        )

        for doc_chunks in final_results:
            doc_chunks[:] = [
                chunk for chunk in doc_chunks if chunk["text"] and chunk["text"].strip()
            ]

            for chunk in doc_chunks:
                if chunk["token_count"] > self.MAX_TOKENS_PER_CHUNK:
                    raise SyncFailureError(
                        f"PROGRAMMING ERROR: Chunk has {chunk['token_count']} tokens "
                        f"after TokenChunker fallback (max: {self.MAX_TOKENS_PER_CHUNK}). "
                    )

        return final_results

    def _recount_tokens_with_tiktoken(self, semantic_results: List[List[Any]]) -> List[List[Any]]:
        """Recount all chunks with tiktoken cl100k_base for OpenAI compatibility."""
        for chunks in semantic_results:
            for chunk in chunks:
                chunk.token_count = len(
                    self._tiktoken_tokenizer.encode(chunk.text, allowed_special="all")
                )
        return semantic_results

    def _apply_safety_net_batched(
        self, semantic_results: List[List[Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Split oversized chunks using TokenChunker fallback."""
        oversized_texts = []
        oversized_map = {}

        for doc_idx, chunks in enumerate(semantic_results):
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.token_count > self.MAX_TOKENS_PER_CHUNK:
                    pos = len(oversized_texts)
                    oversized_texts.append(chunk.text)
                    oversized_map[pos] = (doc_idx, chunk_idx)

        split_results_by_position = {}
        if oversized_texts:
            split_results = self._token_chunker.chunk_batch(oversized_texts)
            split_results_by_position = dict(enumerate(split_results))

        final_results = []
        for doc_idx, chunks in enumerate(semantic_results):
            final_chunks = []
            for chunk_idx, chunk in enumerate(chunks):
                oversized_pos = next(
                    (
                        pos
                        for pos, (d_idx, c_idx) in oversized_map.items()
                        if d_idx == doc_idx and c_idx == chunk_idx
                    ),
                    None,
                )

                if oversized_pos is not None:
                    split_chunks = split_results_by_position[oversized_pos]
                    for sub_chunk in split_chunks:
                        final_chunks.append(self._convert_chunk(sub_chunk))
                else:
                    final_chunks.append(self._convert_chunk(chunk))

            final_results.append(final_chunks)

        return final_results

    def _convert_chunk(self, chunk) -> Dict[str, Any]:
        """Convert Chonkie Chunk object to dict format."""
        return {
            "text": chunk.text,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
        }
```

## Embedding

Once the text has been chunked, the next step is to embed the chunks. Embedding is the process of converting text into a numerical representation (a vector) that can be used by machine learning models.

We use two types of embeddings:

1.  **Dense Embeddings**: These are high-dimensional vectors that capture the semantic meaning of the text. We use a `DenseEmbedder` to generate these embeddings.
2.  **Sparse Embeddings**: These are high-dimensional vectors that are mostly zero. They are used to capture the keyword-based information in the text. We use a `SparseEmbedder` to generate these embeddings.

### Qdrant Chunk Embed Processor

The `QdrantChunkEmbedProcessor` is responsible for orchestrating the entire chunking and embedding pipeline. It takes a list of entities, chunks them, and then computes the dense and sparse embeddings for each chunk.

```python
"""Chunk and embed processor for vector databases requiring pre-computed embeddings.

Used by: Qdrant, Pinecone, and similar vector DBs.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from airweave.platform.entities._base import BaseEntity, CodeFileEntity
from airweave.platform.sync.exceptions import SyncFailureError
from airweave.platform.sync.pipeline.text_builder import text_builder
from airweave.platform.sync.processors.protocol import ContentProcessor
from airweave.platform.sync.processors.utils import filter_empty_representations

if TYPE_CHECKING:
    from airweave.platform.contexts import SyncContext


class QdrantChunkEmbedProcessor(ContentProcessor):
    """Processor that chunks text and computes embeddings for Qdrant.

    Pipeline:
    1. Build textual representation (text extraction from files/web)
    2. Chunk text (semantic for text, AST for code)
    3. Compute embeddings (dense + sparse)

    Output:
        Chunk entities with:
        - entity_id: "{original_id}__chunk_{idx}"
        - textual_representation: chunk text
        - airweave_system_metadata.vectors: [dense_vector, sparse_vector]
        - airweave_system_metadata.original_entity_id: original entity_id
        - airweave_system_metadata.chunk_index: chunk position
    """

    async def process(
        self,
        entities: List[BaseEntity],
        sync_context: "SyncContext",
    ) -> List[BaseEntity]:
        """Process entities through full chunk+embed pipeline."""
        if not entities:
            return []

        # Step 1: Build textual representations
        processed = await text_builder.build_for_batch(entities, sync_context)

        # Step 2: Filter empty representations
        processed = await filter_empty_representations(processed, sync_context, "ChunkEmbed")
        if not processed:
            sync_context.logger.debug("[QdrantChunkEmbedProcessor] No entities after text building")
            return []

        # Step 3: Chunk entities
        chunk_entities = await self._chunk_entities(processed, sync_context)

        # Step 4: Release parent text (memory optimization)
        for entity in processed:
            entity.textual_representation = None

        # Step 5: Embed chunks
        await self._embed_entities(chunk_entities, sync_context)

        sync_context.logger.debug(
            f"[QdrantChunkEmbedProcessor] {len(entities)} entities → {len(chunk_entities)} chunks"
        )

        return chunk_entities

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------

    async def _chunk_entities(
        self,
        entities: List[BaseEntity],
        sync_context: "SyncContext",
    ) -> List[BaseEntity]:
        """Route entities to appropriate chunker."""
        code_entities = [e for e in entities if isinstance(e, CodeFileEntity)]
        textual_entities = [e for e in entities if not isinstance(e, CodeFileEntity)]

        all_chunks: List[BaseEntity] = []

        if code_entities:
            chunks = await self._chunk_code_entities(code_entities, sync_context)
            all_chunks.extend(chunks)

        if textual_entities:
            chunks = await self._chunk_textual_entities(textual_entities, sync_context)
            all_chunks.extend(chunks)

        return all_chunks

    async def _chunk_code_entities(
        self,
        entities: List[BaseEntity],
        sync_context: "SyncContext",
    ) -> List[BaseEntity]:
        """Chunk code with AST-aware CodeChunker."""
        from airweave.platform.chunkers.code import CodeChunker

        # Filter unsupported languages
        supported, unsupported = await self._filter_unsupported_languages(entities)
        if unsupported:
            # TODO: Record skipped entities through exception handling
            await sync_context.entity_tracker.record_skipped(len(unsupported))

        if not supported:
            return []

        chunker = CodeChunker()
        texts = [e.textual_representation for e in supported]

        try:
            chunk_lists = await chunker.chunk_batch(texts)
        except Exception as e:
            raise SyncFailureError(f"[QdrantChunkEmbedProcessor] CodeChunker failed: {e}")

        return self._multiply_entities(supported, chunk_lists, sync_context)

    async def _chunk_textual_entities(
        self,
        entities: List[BaseEntity],
        sync_context: "SyncContext",
    ) -> List[BaseEntity]:
        """Chunk text with SemanticChunker."""
        from airweave.platform.chunkers.semantic import SemanticChunker

        chunker = SemanticChunker()
        texts = [e.textual_representation for e in entities]

        try:
            chunk_lists = await chunker.chunk_batch(texts)
        except Exception as e:
            raise SyncFailureError(f"[QdrantChunkEmbedProcessor] SemanticChunker failed: {e}")

        return self._multiply_entities(entities, chunk_lists, sync_context)

    async def _filter_unsupported_languages(
        self,
        entities: List[BaseEntity],
    ) -> Tuple[List[BaseEntity], List[BaseEntity]]:
        """Filter code entities by tree-sitter support."""
        try:
            from magika import Magika
            from tree_sitter_language_pack import get_parser
        except ImportError:
            return entities, []

        magika = Magika()
        supported: List[BaseEntity] = []
        unsupported: List[BaseEntity] = []

        for entity in entities:
            try:
                text_bytes = entity.textual_representation.encode("utf-8")
                result = magika.identify_bytes(text_bytes)
                lang = result.output.label.lower()
                get_parser(lang)
                supported.append(entity)
            except (LookupError, Exception):
                unsupported.append(entity)

        return supported, unsupported

    def _multiply_entities(
        self,
        entities: List[BaseEntity],
        chunk_lists: List[List[Dict[str, Any]]],
        sync_context: "SyncContext",
    ) -> List[BaseEntity]:
        """Create chunk entities from chunker output."""
        chunk_entities: List[BaseEntity] = []

        for entity, chunks in zip(entities, chunk_lists, strict=True):
            if not chunks:
                continue

            original_id = entity.entity_id

            for idx, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "")
                if not chunk_text or not chunk_text.strip():
                    continue

                chunk_entity = entity.model_copy(deep=True)
                chunk_entity.textual_representation = chunk_text
                chunk_entity.entity_id = f"{original_id}__chunk_{idx}"
                chunk_entity.airweave_system_metadata.chunk_index = idx
                chunk_entity.airweave_system_metadata.original_entity_id = original_id

                chunk_entities.append(chunk_entity)

        return chunk_entities

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    async def _embed_entities(
        self,
        chunk_entities: List[BaseEntity],
        sync_context: "SyncContext",
    ) -> None:
        """Compute dense and sparse embeddings.

        This processor is only used for CHUNKS_AND_EMBEDDINGS destinations,
        which always need both dense and sparse embeddings for hybrid search.
        """
        if not chunk_entities:
            return

        from airweave.platform.embedders import DenseEmbedder, SparseEmbedder

        # Dense embeddings
        dense_texts = [e.textual_representation for e in chunk_entities]
        dense_embedder = DenseEmbedder(vector_size=sync_context.collection.vector_size)
        dense_embeddings = await dense_embedder.embed_many(dense_texts, sync_context)

        # Sparse embeddings for hybrid search
        sparse_texts = [
            json.dumps(
                e.model_dump(mode="json", exclude={"airweave_system_metadata"}),
                sort_keys=True,
            )
            for e in chunk_entities
        ]
        sparse_embedder = SparseEmbedder()
        sparse_embeddings = await sparse_embedder.embed_many(sparse_texts, sync_context)

        # Assign vectors to entities
        for i, entity in enumerate(chunk_entities):
            entity.airweave_system_metadata.vectors = [dense_embeddings[i], sparse_embeddings[i]]

        # Validate
        for entity in chunk_entities:
            if not entity.airweave_system_metadata.vectors:
                raise SyncFailureError(f"Entity {entity.entity_id} has no vectors")
            if entity.airweave_system_metadata.vectors[0] is None:
                raise SyncFailureError(f"Entity {entity.entity_id} has no dense vector")
```

## Retrieval

Once the chunks have been embedded, they are stored in a vector database like Qdrant. The retrieval process involves querying this database to find the most relevant chunks for a given query.

### Hybrid Search

We use a technique called **hybrid search**, which combines the results of both dense and sparse vector search. This allows us to get the best of both worlds:

*   **Dense search** is good at finding semantically similar chunks, even if they don't share any keywords with the query.
*   **Sparse search** is good at finding chunks that contain the exact keywords from the query.

By combining the results of these two search methods, we can achieve better retrieval performance than either method could on its own.

The basic process for retrieval is as follows:

1.  **Query Embedding**: When a query comes in, it is embedded using the same `DenseEmbedder` and `SparseEmbedder` that were used for the chunks.
2.  **Vector Search**: The query vectors are then used to search the vector database. The database will return a list of the most similar chunks for both the dense and sparse vectors.
3.  **Result Combination**: The results from the dense and sparse searches are then combined and re-ranked to produce the final list of chunks that will be returned to the user.
