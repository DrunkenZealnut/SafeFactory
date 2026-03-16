"""Offline Knowledge Graph builder — extracts entities and relations from Pinecone chunks."""

import json
import logging
import time

from models import db, KGEntity, KGRelation, KGEntityChunk
from services.graph_config import get_graph_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """너는 산업안전 도메인의 Knowledge Graph 구축 전문가야.
주어진 텍스트 청크에서 엔티티(개체)와 관계를 추출해.

## 엔티티 타입
{entity_types}

## 관계 타입
uses, part_of, causes, prevents, related_to, requires

## 규칙
1. 엔티티명은 한국어 기준으로 정규화 (예: "CVD" → "CVD 공정")
2. 동일 개념의 영어/한국어 표현은 aliases에 포함
3. 관계는 (source, relation_type, target) 트리플로 추출
4. 확실하지 않은 관계는 confidence를 낮게 설정 (0.0~1.0)
5. 각 엔티티에 1-2문장 설명 생성
6. 텍스트에 명확히 나타나는 엔티티만 추출 (추측 금지)

## JSON 출력 형식 (반드시 이 형식만 출력)
{{
  "entities": [
    {{"name": "CVD 공정", "type": "공정", "description": "화학기상증착...", "aliases": ["Chemical Vapor Deposition"]}}
  ],
  "relations": [
    {{"source": "CVD 공정", "type": "uses", "target": "실란", "confidence": 0.9}}
  ]
}}"""


def _build_user_prompt(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        cid = c.get('id', f'chunk_{i}')
        text = c.get('content', c.get('metadata', {}).get('content', ''))[:1500]
        parts.append(f"[Chunk {i} | id={cid}]\n{text}")
    return "아래 텍스트 청크들에서 엔티티와 관계를 추출해줘.\n\n" + "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().replace(' ', '').strip()


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Builds a Knowledge Graph from Pinecone chunks using LLM extraction."""

    RELATION_TYPES = {'uses', 'part_of', 'causes', 'prevents', 'related_to', 'requires'}

    def __init__(self, namespace: str, gemini_client=None):
        self.namespace = namespace
        cfg = get_graph_config(namespace)
        self.entity_types = cfg.get('entity_types', [])
        if gemini_client is None:
            from services.singletons import get_gemini_client
            gemini_client = get_gemini_client()
        self.gemini = gemini_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict], batch_size: int = 20) -> dict:
        """Run the full extraction pipeline on the provided chunks.

        Args:
            chunks: list of dicts with at least 'id' and 'content' (or nested metadata.content).
            batch_size: number of chunks per LLM call.

        Returns:
            dict with 'entities', 'relations', 'entity_chunks' counts.
        """
        total_entities = 0
        total_relations = 0
        total_mappings = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                extracted = self._extract_batch(batch)
                counts = self._save_extracted(extracted, batch)
                total_entities += counts['entities']
                total_relations += counts['relations']
                total_mappings += counts['mappings']
                logger.info(
                    "[GraphBuilder] Batch %d-%d: %d entities, %d relations",
                    i, i + len(batch), counts['entities'], counts['relations'],
                )
            except Exception:
                logger.exception("[GraphBuilder] Batch %d-%d failed, skipping", i, i + len(batch))

        # Normalize (merge duplicates)
        merged = self._normalize_entities()
        logger.info("[GraphBuilder] Merged %d duplicate entities", merged)

        stats = {
            'entities': db.session.query(KGEntity).filter_by(namespace=self.namespace).count(),
            'relations': db.session.query(KGRelation).filter_by(namespace=self.namespace).count(),
            'entity_chunks': db.session.query(KGEntityChunk).filter_by(namespace=self.namespace).count(),
        }
        logger.info("[GraphBuilder] Final stats for %s: %s", self.namespace, stats)
        return stats

    def reset(self):
        """Delete all graph data for this namespace."""
        KGEntityChunk.query.filter_by(namespace=self.namespace).delete()
        KGRelation.query.filter_by(namespace=self.namespace).delete()
        KGEntity.query.filter_by(namespace=self.namespace).delete()
        db.session.commit()
        logger.info("[GraphBuilder] Reset graph for namespace=%s", self.namespace)

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def _extract_batch(self, chunks: list[dict]) -> dict:
        """Call Gemini Flash to extract entities/relations from a batch."""
        system = _SYSTEM_PROMPT.format(entity_types=', '.join(self.entity_types))
        user = _build_user_prompt(chunks)

        response = self.gemini.models.generate_content(
            model='gemini-2.0-flash',
            contents=[{'role': 'user', 'parts': [{'text': system + '\n\n' + user}]}],
            config={'response_mime_type': 'application/json', 'temperature': 0.1},
        )
        text = response.text.strip()
        # Strip markdown fences if present
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return json.loads(text)

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _save_extracted(self, extracted: dict, chunks: list[dict]) -> dict:
        """Persist extracted entities/relations into SQLite."""
        entities_data = extracted.get('entities', [])
        relations_data = extracted.get('relations', [])

        entity_name_map: dict[str, KGEntity] = {}
        new_entities = 0
        new_relations = 0
        new_mappings = 0

        # --- Entities ---
        for e in entities_data:
            name = e.get('name', '').strip()
            if not name:
                continue
            norm = _normalize(name)
            etype = e.get('type', '').strip()
            if etype and etype not in self.entity_types:
                etype = self.entity_types[0] if self.entity_types else 'unknown'

            existing = db.session.query(KGEntity).filter_by(
                name_normalized=norm, namespace=self.namespace,
            ).first()

            if existing:
                # Update description if richer
                if e.get('description') and (not existing.description or len(e['description']) > len(existing.description)):
                    existing.description = e['description']
                # Merge aliases
                old_aliases = set(existing.aliases)
                new_aliases = set(e.get('aliases', []))
                if new_aliases - old_aliases:
                    existing.aliases = list(old_aliases | new_aliases)
                entity_name_map[norm] = existing
            else:
                ent = KGEntity(
                    name=name,
                    name_normalized=norm,
                    entity_type=etype or 'unknown',
                    namespace=self.namespace,
                    description=e.get('description', ''),
                )
                ent.aliases = e.get('aliases', [])
                db.session.add(ent)
                db.session.flush()  # get id
                entity_name_map[norm] = ent
                new_entities += 1

        # --- Relations ---
        for r in relations_data:
            src_norm = _normalize(r.get('source', ''))
            tgt_norm = _normalize(r.get('target', ''))
            rtype = r.get('type', 'related_to')
            if rtype not in self.RELATION_TYPES:
                rtype = 'related_to'

            src_ent = entity_name_map.get(src_norm)
            tgt_ent = entity_name_map.get(tgt_norm)
            if not src_ent or not tgt_ent or src_ent.id == tgt_ent.id:
                continue

            existing_rel = db.session.query(KGRelation).filter_by(
                source_id=src_ent.id, target_id=tgt_ent.id,
                relation_type=rtype, namespace=self.namespace,
            ).first()
            if not existing_rel:
                rel = KGRelation(
                    source_id=src_ent.id, target_id=tgt_ent.id,
                    relation_type=rtype,
                    confidence=float(r.get('confidence', 0.8)),
                    namespace=self.namespace,
                )
                db.session.add(rel)
                new_relations += 1

        # --- Entity-Chunk mappings ---
        chunk_id_set = {c.get('id', '') for c in chunks if c.get('id')}
        for norm, ent in entity_name_map.items():
            for cid in chunk_id_set:
                existing_ec = db.session.query(KGEntityChunk).filter_by(
                    entity_id=ent.id, chunk_vector_id=cid, namespace=self.namespace,
                ).first()
                if not existing_ec:
                    db.session.add(KGEntityChunk(
                        entity_id=ent.id,
                        chunk_vector_id=cid,
                        relevance_score=1.0,
                        namespace=self.namespace,
                    ))
                    new_mappings += 1

        db.session.commit()
        return {'entities': new_entities, 'relations': new_relations, 'mappings': new_mappings}

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _normalize_entities(self) -> int:
        """Merge entities with the same name_normalized within the namespace."""
        from sqlalchemy import func
        dupes = (
            db.session.query(KGEntity.name_normalized, func.count(KGEntity.id))
            .filter_by(namespace=self.namespace)
            .group_by(KGEntity.name_normalized)
            .having(func.count(KGEntity.id) > 1)
            .all()
        )
        merged = 0
        for norm, _cnt in dupes:
            entities = (
                db.session.query(KGEntity)
                .filter_by(name_normalized=norm, namespace=self.namespace)
                .order_by(KGEntity.id)
                .all()
            )
            primary = entities[0]
            for dup in entities[1:]:
                # Move relations
                db.session.query(KGRelation).filter_by(source_id=dup.id).update({'source_id': primary.id})
                db.session.query(KGRelation).filter_by(target_id=dup.id).update({'target_id': primary.id})
                # Move chunk mappings
                db.session.query(KGEntityChunk).filter_by(entity_id=dup.id).update({'entity_id': primary.id})
                # Merge aliases
                all_aliases = set(primary.aliases) | set(dup.aliases)
                primary.aliases = list(all_aliases)
                # Merge description
                if dup.description and (not primary.description or len(dup.description) > len(primary.description)):
                    primary.description = dup.description
                db.session.delete(dup)
                merged += 1
        db.session.commit()
        return merged
