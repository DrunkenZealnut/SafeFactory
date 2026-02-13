"""
Semantic Chunker Module
Implements semantic chunking for intelligent text splitting.
"""

import json
import os
import re
import certifi
import httpx
import tiktoken
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI

# Set SSL certificate environment variables
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    index: int
    source_file: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Optional[Dict] = None

    # Contextual chunking fields
    document_title: Optional[str] = None
    document_summary: Optional[str] = None
    section_title: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: Optional[List[str]] = None

    # Page tracking fields
    page_id: Optional[int] = None
    page_id_end: Optional[int] = None


def build_page_line_map(markdown_text: str, meta: Dict) -> Dict[int, int]:
    """
    Build a line_number → page_id mapping using anchors from _meta.json.

    Anchors are collected from:
    1. Image references: _page_N_*.jpeg in markdown → page_id=N (100% accurate)
    2. TOC heading matching: meta['table_of_contents'] titles → markdown headings (95%+ accurate)
    3. Interpolation: lines between anchors are distributed by page_stats block counts

    Args:
        markdown_text: Full markdown text
        meta: Parsed _meta.json dict with 'table_of_contents' and 'page_stats'

    Returns:
        Dict mapping 0-based line index → page_id
    """
    lines = markdown_text.split('\n')
    total_lines = len(lines)
    if total_lines == 0:
        return {}

    page_stats = meta.get('page_stats', [])
    toc = meta.get('table_of_contents', [])

    # Build page_id → block_count map from page_stats
    page_block_counts: Dict[int, int] = {}
    max_page_id = -1
    for ps in page_stats:
        pid = ps.get('page_id', -1)
        if pid < 0:
            continue
        # Sum all block counts for this page
        blocks = 0
        for block_type, count in ps.get('block_counts', []):
            if block_type != 'BlockType':
                blocks += count
        page_block_counts[pid] = max(blocks, 1)
        max_page_id = max(max_page_id, pid)

    if max_page_id < 0:
        return {}

    # --- Collect anchors: (line_index, page_id) ---
    anchors: List[Tuple[int, int]] = []

    # 1. Image anchors: look for _page_N_ pattern in image references
    img_pattern = re.compile(r'_page_(\d+)_')
    for i, line in enumerate(lines):
        m = img_pattern.search(line)
        if m:
            page_id = int(m.group(1))
            if page_id <= max_page_id:
                anchors.append((i, page_id))

    # 2. TOC heading anchors: match TOC titles to markdown headings
    if toc:
        # Build heading → line_index map from markdown
        heading_pattern = re.compile(r'^#{1,6}\s+(.+)$')
        md_headings: List[Tuple[int, str]] = []
        for i, line in enumerate(lines):
            hm = heading_pattern.match(line.strip())
            if hm:
                md_headings.append((i, hm.group(1).strip()))

        # Match TOC entries to markdown headings
        for toc_entry in toc:
            toc_title = toc_entry.get('title', '').strip()
            toc_page = toc_entry.get('page_id')
            if not toc_title or toc_page is None:
                continue

            # Find matching heading in markdown (exact or substring match)
            for line_idx, heading_text in md_headings:
                # Normalize for comparison: strip whitespace, compare core text
                if _normalize_title(toc_title) == _normalize_title(heading_text):
                    anchors.append((line_idx, toc_page))
                    break

    # Deduplicate and sort anchors by line index
    # If multiple anchors on same line, keep the one with higher page_id (more specific)
    anchor_dict: Dict[int, int] = {}
    for line_idx, page_id in anchors:
        if line_idx not in anchor_dict or page_id > anchor_dict[line_idx]:
            anchor_dict[line_idx] = page_id

    sorted_anchors = sorted(anchor_dict.items(), key=lambda x: x[0])

    if not sorted_anchors:
        # No anchors found: distribute all lines proportionally across all pages
        return _distribute_lines_by_blocks(0, total_lines - 1, 0, max_page_id, page_block_counts)

    # --- Interpolate between anchors ---
    page_map: Dict[int, int] = {}

    # Region before first anchor
    first_anchor_line, first_anchor_page = sorted_anchors[0]
    if first_anchor_line > 0 and first_anchor_page > 0:
        region = _distribute_lines_by_blocks(
            0, first_anchor_line - 1,
            0, first_anchor_page - 1,
            page_block_counts
        )
        page_map.update(region)
    elif first_anchor_line > 0:
        # First anchor is page 0, assign all preceding lines to page 0
        for i in range(first_anchor_line):
            page_map[i] = 0

    # Between consecutive anchors
    for idx in range(len(sorted_anchors)):
        anchor_line, anchor_page = sorted_anchors[idx]
        page_map[anchor_line] = anchor_page

        if idx + 1 < len(sorted_anchors):
            next_line, next_page = sorted_anchors[idx + 1]

            if next_line > anchor_line + 1:
                if next_page > anchor_page:
                    # Interpolate between anchors
                    region = _distribute_lines_by_blocks(
                        anchor_line + 1, next_line - 1,
                        anchor_page, next_page - 1,
                        page_block_counts
                    )
                    page_map.update(region)
                else:
                    # Same or lower page (shouldn't happen often), fill with anchor_page
                    for i in range(anchor_line + 1, next_line):
                        page_map[i] = anchor_page

    # Region after last anchor
    last_anchor_line, last_anchor_page = sorted_anchors[-1]
    if last_anchor_line < total_lines - 1:
        if last_anchor_page < max_page_id:
            region = _distribute_lines_by_blocks(
                last_anchor_line + 1, total_lines - 1,
                last_anchor_page, max_page_id,
                page_block_counts
            )
            page_map.update(region)
        else:
            # Last anchor is already max page, assign remaining to max page
            for i in range(last_anchor_line + 1, total_lines):
                page_map[i] = max_page_id

    return page_map


def _char_offset_to_line(char_offset: int, line_starts: List[int]) -> int:
    """Convert a character offset to a 0-based line index using binary search."""
    lo, hi = 0, len(line_starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if line_starts[mid] <= char_offset:
            lo = mid + 1
        else:
            hi = mid - 1
    return max(0, lo - 1)


def _normalize_title(title: str) -> str:
    """Normalize a title for comparison by removing extra whitespace and lowering."""
    return re.sub(r'\s+', ' ', title.strip()).lower()


def _distribute_lines_by_blocks(
    start_line: int,
    end_line: int,
    start_page: int,
    end_page: int,
    page_block_counts: Dict[int, int]
) -> Dict[int, int]:
    """
    Distribute lines between start_line..end_line across pages start_page..end_page
    proportionally based on block counts from page_stats.

    Returns:
        Dict mapping line_index → page_id
    """
    result: Dict[int, int] = {}
    total_lines = end_line - start_line + 1

    if total_lines <= 0:
        return result

    if start_page > end_page:
        for i in range(start_line, end_line + 1):
            result[i] = start_page
        return result

    if start_page == end_page:
        for i in range(start_line, end_line + 1):
            result[i] = start_page
        return result

    # Collect block counts for pages in range
    pages = list(range(start_page, end_page + 1))
    blocks = [page_block_counts.get(p, 1) for p in pages]
    total_blocks = sum(blocks)

    if total_blocks == 0:
        total_blocks = len(pages)
        blocks = [1] * len(pages)

    # Distribute lines proportionally
    current_line = start_line
    for i, page_id in enumerate(pages):
        if i == len(pages) - 1:
            # Last page gets remaining lines
            line_count = end_line - current_line + 1
        else:
            line_count = max(1, round(total_lines * blocks[i] / total_blocks))

        for j in range(line_count):
            line_idx = current_line + j
            if line_idx > end_line:
                break
            result[line_idx] = page_id

        current_line += line_count
        if current_line > end_line:
            break

    # Fill any remaining unassigned lines with end_page
    for i in range(start_line, end_line + 1):
        if i not in result:
            result[i] = end_page

    return result


class SemanticChunker:
    """
    Splits text into semantically meaningful chunks.
    Uses a combination of structural analysis and embedding similarity.

    Enhanced with contextual chunking to preserve document context.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "text-embedding-3-small",
        max_chunk_tokens: int = 500,
        min_chunk_tokens: int = 100,
        overlap_tokens: int = 50,
        similarity_threshold: float = 0.5,
        enable_contextual: bool = True
    ):
        """
        Initialize the SemanticChunker.

        Args:
            openai_api_key: OpenAI API key for embeddings
            model: Embedding model to use
            max_chunk_tokens: Maximum tokens per chunk
            min_chunk_tokens: Minimum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            similarity_threshold: Threshold for semantic similarity
            enable_contextual: Enable contextual chunking with document metadata
        """
        # Create httpx client with explicit SSL certificate verification
        http_client = httpx.Client(verify=certifi.where())
        self.client = OpenAI(api_key=openai_api_key, http_client=http_client)
        self.model = model
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold
        self.enable_contextual = enable_contextual
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken for accurate measurement."""
        return len(self.encoding.encode(text))

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common sentence endings
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_by_structure(self, text: str) -> List[str]:
        """Split text by structural elements (headers, paragraphs, etc.)."""
        # Split by markdown headers
        header_pattern = r'^#{1,6}\s+.+$'

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)

        segments = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if it's a header
            if re.match(header_pattern, para, re.MULTILINE):
                segments.append(para)
            else:
                # Further split long paragraphs into sentences
                if self._count_tokens(para) > self.max_chunk_tokens:
                    sentences = self._split_into_sentences(para)
                    segments.extend(sentences)
                else:
                    segments.append(para)

        return segments

    # NCS section boundary patterns for hard splitting
    NCS_SECTION_PATTERNS = [
        r'^#{1,3}\s+학습\s*\d+',        # 학습 단위 헤더
        r'^#{1,3}\s+\d+-\d+\.',          # 하위 섹션 (예: "1-1. 시장 동향 분석")
        r'^\|?\s*필요\s*지식',           # 필요 지식 (표 형식 포함)
        r'^\|?\s*수행\s*내용',           # 수행 내용
        r'^\|?\s*학습\s*목표',           # 학습 목표
        r'^#{1,3}\s+평가',               # 평가
        r'^#{1,3}\s+교수',               # 교수·학습 방법
        r'^\|?\s*안전.*유의',            # 안전·유의사항
        r'^\|?\s*핵심\s*용어',           # 핵심 용어
    ]

    def _split_by_ncs_structure(self, text: str) -> List[str]:
        """Split NCS text by logical section boundaries then by structure within each section."""
        # Compile NCS boundary pattern
        boundary_pattern = '|'.join(self.NCS_SECTION_PATTERNS)
        compiled = re.compile(boundary_pattern, re.MULTILINE)

        # Find all NCS section boundaries
        boundaries = [m.start() for m in compiled.finditer(text)]

        if not boundaries:
            # Fallback to default structure splitting
            return self._split_by_structure(text)

        # Add start and end
        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        boundaries.append(len(text))

        # Split text into NCS sections
        sections = []
        for i in range(len(boundaries) - 1):
            section_text = text[boundaries[i]:boundaries[i + 1]].strip()
            if section_text:
                sections.append(section_text)

        # Within each NCS section, apply default structure splitting
        all_segments = []
        for section in sections:
            sub_segments = self._split_by_structure(section)
            all_segments.extend(sub_segments)

        return all_segments

    def _merge_small_segments(self, segments: List[str]) -> List[str]:
        """Merge segments that are too small."""
        merged = []
        current = ""

        # Reserve tokens for contextual prefix (e.g., "[문서: ... | 섹션: ...]")
        effective_max = self.max_chunk_tokens - 50 if self.enable_contextual else self.max_chunk_tokens

        for segment in segments:
            if not segment:
                continue

            test_combined = f"{current}\n\n{segment}" if current else segment
            combined_tokens = self._count_tokens(test_combined)

            if combined_tokens <= effective_max:
                current = test_combined
            else:
                if current:
                    merged.append(current)
                current = segment

        if current:
            merged.append(current)

        return merged

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks for context continuity."""
        if len(chunks) <= 1:
            return chunks

        overlapped = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Get last portion of previous chunk by tokens
                prev_chunk = chunks[i-1]
                prev_tokens = self.encoding.encode(prev_chunk)

                if len(prev_tokens) > self.overlap_tokens:
                    overlap_text = self.encoding.decode(prev_tokens[-self.overlap_tokens:])
                    overlapped.append(f"...{overlap_text}\n\n{chunk}")
                else:
                    overlapped.append(chunk)

        return overlapped

    def _extract_document_title(self, text: str) -> Optional[str]:
        """Extract document title from the first header or line."""
        lines = text.strip().split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            # Check for markdown headers
            if line.startswith('#'):
                return line.lstrip('#').strip()
            # Check for non-empty line as potential title
            if line and len(line) < 100:
                return line
        return None

    def _extract_section_titles(self, text: str) -> Dict[int, str]:
        """Extract section titles with their character positions."""
        sections = {}
        header_pattern = r'^(#{1,6})\s+(.+)$'

        for match in re.finditer(header_pattern, text, re.MULTILINE):
            sections[match.start()] = match.group(2).strip()

        return sections

    def _generate_document_summary(self, text: str, max_length: int = 300) -> str:
        """Generate a brief summary of the document.

        Skips image references, table markup, and other non-content lines
        to find the first meaningful paragraph.
        """
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            # Skip headers
            if para.startswith('#'):
                continue
            # Skip image references
            if para.startswith('![]') or para.startswith('!['):
                continue
            # Skip table rows (starts with | or contains mostly |)
            if para.startswith('|') or para.count('|') > 3:
                continue
            # Skip HTML comments (page markers etc.)
            if para.startswith('<!--'):
                continue
            # Skip very short or whitespace-only paragraphs
            if len(para) < 30:
                continue
            # Skip lines that are just formatting (---, ===, etc.)
            if re.match(r'^[-=*_\s]+$', para):
                continue
            # Return first substantial paragraph
            if len(para) > max_length:
                return para[:max_length] + "..."
            return para

        # Fallback: collect header texts as summary
        headers = []
        for para in paragraphs:
            para = para.strip()
            if para.startswith('#'):
                header_text = re.sub(r'^#+\s*', '', para).strip()
                if header_text and len(header_text) > 5:
                    headers.append(header_text)
            if len(' | '.join(headers)) > max_length:
                break
        if headers:
            return ' | '.join(headers)[:max_length]

        return text[:max_length].strip() + "..." if len(text) > max_length else text.strip()

    def _find_section_for_position(self, position: int, sections: Dict[int, str]) -> Optional[str]:
        """Find the section title for a given character position."""
        current_section = None
        for sec_pos, sec_title in sorted(sections.items()):
            if sec_pos <= position:
                current_section = sec_title
            else:
                break
        return current_section

    def _extract_ncs_metadata(self, source_file: str, text: str) -> Dict:
        """Extract NCS-specific metadata from file path and content."""
        import unicodedata
        ncs_meta = {}

        # Normalize path to NFC for consistent Korean character matching (macOS uses NFD)
        normalized_path = unicodedata.normalize('NFC', source_file)

        # Extract NCS code from filename
        # Matches both versioned (LM1903060101_23v6) and unversioned (LM1903060107)
        code_match = re.search(r'(LM\d{10}(?:_\d+v\d+)?)', normalized_path)
        if code_match:
            ncs_meta['ncs_code'] = code_match.group(1)

        # Extract NCS category from path
        ncs_categories = ['반도체개발', '반도체장비', '반도체재료', '반도체제조']
        for cat in ncs_categories:
            if cat in normalized_path:
                ncs_meta['ncs_category'] = cat
                break

        # Extract document title from path (handles both versioned and unversioned)
        title_match = re.search(r'LM\d{10}(?:_\d+v\d+)?_(.+?)(?:/|$)', normalized_path)
        if title_match:
            ncs_meta['ncs_document_title'] = title_match.group(1).replace('_', ' ')

        return ncs_meta

    def _classify_ncs_section(self, section_title: str) -> str:
        """Classify NCS section by type based on title patterns.

        Strips markdown bold markers and normalizes special characters
        before matching, to handle variants like **필요 지식 /**, 필요 지식 🖊, etc.
        """
        if not section_title:
            return 'general'

        # Strip markdown bold and extra whitespace
        title = re.sub(r'\*+', '', section_title).strip()
        # Normalize middle-dot variants (・, ‧, ·) to a single space
        title = re.sub(r'[・‧·]', ' ', title)

        patterns = [
            (r'필요\s*지식', 'required_knowledge'),
            (r'수행\s*순서', 'performance_procedure'),
            (r'수행\s*내용', 'performance_content'),
            (r'수행\s*tip', 'performance_tip'),
            (r'학습\s*목표', 'learning_objective'),
            (r'학습모듈의\s*목표', 'learning_objective'),
            (r'평가\s*준거', 'evaluation_criteria'),
            (r'평가\s*방법', 'evaluation_method'),
            (r'평가자\s*체크리스트', 'evaluation_checklist'),
            (r'평가자\s*질문', 'evaluation_question'),
            (r'서술형\s*시험', 'evaluation_written'),
            (r'논술형\s*시험', 'evaluation_essay'),
            (r'구두\s*발표', 'evaluation_oral'),
            (r'피드백', 'feedback'),
            (r'교수\s*방법', 'teaching_method'),
            (r'학습\s*방법', 'learning_method'),
            (r'안전\s*유의\s*사항', 'safety_notes'),
            (r'안전\s*유의사항', 'safety_notes'),
            (r'핵심\s*용어', 'key_terms'),
            (r'선수\s*학습|선수학습', 'prerequisite'),
            (r'기기\s*\(?\s*장비', 'equipment'),
            (r'재료\s*자료', 'materials'),
            (r'학습모듈의\s*내용\s*체계', 'module_structure'),
            (r'NCS\s*학습모듈의\s*위치', 'module_position'),
            (r'NCS\s*학습모듈이란', 'module_intro'),
            (r'학습\s+(\d+)', 'learning_unit'),
        ]

        for pattern, section_type in patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return section_type

        return 'general'

    def _extract_learning_unit(self, section_title: str) -> Optional[int]:
        """Extract learning unit number from section title."""
        if not section_title:
            return None
        match = re.search(r'학습\s*(\d+)', section_title)
        return int(match.group(1)) if match else None

    def _add_contextual_prefix(
        self,
        content: str,
        document_title: Optional[str],
        section_title: Optional[str],
        source_file: str
    ) -> str:
        """Add contextual prefix to chunk content for better embedding."""
        prefix_parts = []

        if document_title:
            prefix_parts.append(f"문서: {document_title}")
        if section_title:
            prefix_parts.append(f"섹션: {section_title}")

        if prefix_parts:
            prefix = " | ".join(prefix_parts)
            return f"[{prefix}]\n\n{content}"

        return content

    def chunk_text(
        self,
        text: str,
        source_file: str,
        use_embeddings: bool = True,
        metadata: Optional[Dict] = None,
        meta_json: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Split text into semantic chunks.

        Args:
            text: Text to chunk
            source_file: Source file path for metadata
            use_embeddings: Whether to use embedding-based similarity
            metadata: Additional metadata to include
            meta_json: Parsed _meta.json from marker conversion (enables page_id assignment)

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        # Build page line map if meta_json is provided
        page_line_map: Optional[Dict[int, int]] = None
        if meta_json and meta_json.get('page_stats'):
            page_line_map = build_page_line_map(text, meta_json)

        # Extract document-level context if contextual chunking is enabled
        document_title = None
        document_summary = None
        section_titles = {}

        if self.enable_contextual:
            document_title = self._extract_document_title(text)
            document_summary = self._generate_document_summary(text)
            section_titles = self._extract_section_titles(text)

        # Detect NCS document and extract metadata
        is_ncs = bool(re.search(r'LM\d{10}', source_file))
        ncs_metadata = self._extract_ncs_metadata(source_file, text) if is_ncs else {}

        # Step 1: Split by structure (NCS-aware or default)
        if is_ncs:
            segments = self._split_by_ncs_structure(text)
        else:
            segments = self._split_by_structure(text)

        if not segments:
            return []

        # Step 2: Merge small segments
        merged_segments = self._merge_small_segments(segments)

        # Step 3: Add overlap
        overlapped_segments = self._add_overlap(merged_segments)

        # Build char_offset → line_number lookup for page mapping
        char_to_line: Optional[List[int]] = None
        if page_line_map:
            # Build cumulative line start positions
            line_starts = [0]
            for line in text.split('\n')[:-1]:
                line_starts.append(line_starts[-1] + len(line) + 1)  # +1 for '\n'
            char_to_line = line_starts

        # Pre-compute segment positions from merged_segments (before overlap)
        segment_positions = []
        pos = 0
        for seg in merged_segments:
            start = text.find(seg[:50], max(0, pos - 100))
            if start == -1:
                start = pos
            segment_positions.append((start, start + len(seg)))
            pos = start + len(seg)

        # Step 4: Create Chunk objects with contextual metadata
        chunks = []
        parent_chunk_id = None

        for i, segment in enumerate(overlapped_segments):
            # Use pre-computed positions from original text (before overlap)
            if i < len(segment_positions):
                start_char, end_char = segment_positions[i]
            else:
                start_char = segment_positions[-1][1] if segment_positions else 0
                end_char = start_char + len(segment)

            # Find section title for this position
            section_title = None
            if self.enable_contextual and section_titles:
                section_title = self._find_section_for_position(start_char, section_titles)

            # Resolve page_id from page_line_map
            chunk_page_id = None
            chunk_page_id_end = None
            if page_line_map and char_to_line:
                start_line = _char_offset_to_line(start_char, char_to_line)
                end_line = _char_offset_to_line(end_char, char_to_line)
                chunk_page_id = page_line_map.get(start_line)
                chunk_page_id_end = page_line_map.get(end_line)
                # If start and end are the same page, no need for page_id_end
                if chunk_page_id_end == chunk_page_id:
                    chunk_page_id_end = None

            # Generate chunk ID
            import hashlib
            chunk_id = hashlib.md5(
                f"{source_file}_{i}_{segment[:50]}".encode()
            ).hexdigest()[:12]

            # Add contextual prefix if enabled
            enhanced_content = segment
            if self.enable_contextual:
                enhanced_content = self._add_contextual_prefix(
                    segment, document_title, section_title, source_file
                )

            chunk_metadata = {
                **(metadata or {}),
                'chunk_index': i,
                'total_chunks': len(overlapped_segments),
                'chunk_id': chunk_id,
                'document_title': document_title,
                'section_title': section_title,
                'has_context': self.enable_contextual,
                'token_count': self._count_tokens(enhanced_content),
            }
            if document_summary and i == 0:
                chunk_metadata['document_summary'] = document_summary[:500]
            if chunk_page_id is not None:
                chunk_metadata['page_id'] = chunk_page_id
            if chunk_page_id_end is not None:
                chunk_metadata['page_id_end'] = chunk_page_id_end

            # Add NCS-specific metadata
            if ncs_metadata:
                chunk_metadata.update(ncs_metadata)
                ncs_section_type = self._classify_ncs_section(section_title)
                chunk_metadata['ncs_section_type'] = ncs_section_type
                learning_unit = self._extract_learning_unit(section_title)
                if learning_unit is not None:
                    chunk_metadata['learning_unit'] = learning_unit

            chunk = Chunk(
                content=enhanced_content,
                index=i,
                source_file=source_file,
                start_char=start_char,
                end_char=end_char,
                token_count=self._count_tokens(enhanced_content),
                metadata=chunk_metadata,
                document_title=document_title,
                document_summary=document_summary if i == 0 else None,  # Only first chunk gets summary
                section_title=section_title,
                parent_chunk_id=parent_chunk_id,
                child_chunk_ids=[],
                page_id=chunk_page_id,
                page_id_end=chunk_page_id_end
            )

            # Set parent-child relationships
            if i > 0 and chunks:
                # Previous chunk is parent of current
                chunk.parent_chunk_id = chunks[-1].metadata.get('chunk_id')
                # Add current as child of previous
                if chunks[-1].child_chunk_ids is None:
                    chunks[-1].child_chunk_ids = []
                chunks[-1].child_chunk_ids.append(chunk_id)

            chunks.append(chunk)

        # Post-process: store parent/child IDs in metadata for Pinecone storage
        for chunk in chunks:
            if chunk.parent_chunk_id:
                chunk.metadata['parent_chunk_id'] = chunk.parent_chunk_id
            if chunk.child_chunk_ids:
                chunk.metadata['child_chunk_ids'] = chunk.child_chunk_ids

        return chunks

    def chunk_json(
        self,
        json_content: str,
        source_file: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk JSON content intelligently.

        Args:
            json_content: JSON string
            source_file: Source file path
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        import json

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError:
            # If invalid JSON, treat as plain text
            return self.chunk_text(json_content, source_file, metadata=metadata)

        # For small JSON, keep as single chunk
        if self._count_tokens(json_content) <= self.max_chunk_tokens:
            return [Chunk(
                content=json_content,
                index=0,
                source_file=source_file,
                start_char=0,
                end_char=len(json_content),
                token_count=self._count_tokens(json_content),
                metadata={
                    **(metadata or {}),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'json_type': type(data).__name__
                }
            )]

        # For large JSON arrays, chunk by items
        if isinstance(data, list):
            return self._chunk_json_array(data, source_file, metadata)

        # For large JSON objects, chunk by keys
        if isinstance(data, dict):
            return self._chunk_json_object(data, source_file, metadata)

        # Fallback to text chunking
        return self.chunk_text(json_content, source_file, metadata=metadata)

    def _chunk_json_array(
        self,
        data: List,
        source_file: str,
        metadata: Optional[Dict]
    ) -> List[Chunk]:
        """Chunk a JSON array."""
        import json

        chunks = []
        current_items = []
        chunk_index = 0

        for item in data:
            item_str = json.dumps(item, ensure_ascii=False, indent=2)
            current_items.append(item)
            current_str = json.dumps(current_items, ensure_ascii=False, indent=2)

            if self._count_tokens(current_str) > self.max_chunk_tokens:
                # Save current chunk without last item
                if len(current_items) > 1:
                    current_items.pop()
                    chunk_content = json.dumps(current_items, ensure_ascii=False, indent=2)
                else:
                    chunk_content = current_str

                chunks.append(Chunk(
                    content=chunk_content,
                    index=chunk_index,
                    source_file=source_file,
                    start_char=0,
                    end_char=len(chunk_content),
                    token_count=self._count_tokens(chunk_content),
                    metadata={
                        **(metadata or {}),
                        'chunk_index': chunk_index,
                        'json_type': 'array_segment'
                    }
                ))

                chunk_index += 1
                current_items = [item] if len(current_items) > 1 else []

        # Add remaining items
        if current_items:
            chunk_content = json.dumps(current_items, ensure_ascii=False, indent=2)
            chunks.append(Chunk(
                content=chunk_content,
                index=chunk_index,
                source_file=source_file,
                start_char=0,
                end_char=len(chunk_content),
                token_count=self._count_tokens(chunk_content),
                metadata={
                    **(metadata or {}),
                    'chunk_index': chunk_index,
                    'json_type': 'array_segment'
                }
            ))

        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)

        return chunks

    def _chunk_json_object(
        self,
        data: Dict,
        source_file: str,
        metadata: Optional[Dict]
    ) -> List[Chunk]:
        """Chunk a JSON object by keys."""
        import json

        chunks = []
        current_obj = {}
        chunk_index = 0

        for key, value in data.items():
            current_obj[key] = value
            current_str = json.dumps(current_obj, ensure_ascii=False, indent=2)

            if self._count_tokens(current_str) > self.max_chunk_tokens:
                # Save current chunk without last key
                if len(current_obj) > 1:
                    del current_obj[key]
                    chunk_content = json.dumps(current_obj, ensure_ascii=False, indent=2)
                else:
                    chunk_content = current_str

                chunks.append(Chunk(
                    content=chunk_content,
                    index=chunk_index,
                    source_file=source_file,
                    start_char=0,
                    end_char=len(chunk_content),
                    token_count=self._count_tokens(chunk_content),
                    metadata={
                        **(metadata or {}),
                        'chunk_index': chunk_index,
                        'json_type': 'object_segment'
                    }
                ))

                chunk_index += 1
                current_obj = {key: value} if len(current_obj) > 1 else {}

        # Add remaining keys
        if current_obj:
            chunk_content = json.dumps(current_obj, ensure_ascii=False, indent=2)
            chunks.append(Chunk(
                content=chunk_content,
                index=chunk_index,
                source_file=source_file,
                start_char=0,
                end_char=len(chunk_content),
                token_count=self._count_tokens(chunk_content),
                metadata={
                    **(metadata or {}),
                    'chunk_index': chunk_index,
                    'json_type': 'object_segment'
                }
            ))

        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)

        return chunks


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        chunker = SemanticChunker(api_key)

        # Test with sample text
        sample_text = """
# Introduction

This is a sample document for testing semantic chunking.

## Section 1

The semantic chunker analyzes text structure and splits it intelligently.
It considers headers, paragraphs, and sentence boundaries.

## Section 2

For JSON content, it can split by array items or object keys.
This ensures that related data stays together in the same chunk.
"""

        chunks = chunker.chunk_text(sample_text, "test.md")
        for chunk in chunks:
            print(f"Chunk {chunk.index}: {chunk.token_count} tokens")
            print(chunk.content[:100])
            print("---")
    else:
        print("OPENAI_API_KEY not found")
