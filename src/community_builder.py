"""Offline community detection and summarization for Knowledge Graph."""

import json
import logging

import networkx as nx

from models import db, KGEntity, KGRelation, KGCommunity, KGCommunityMember
from services.graph_config import get_graph_config

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """다음은 Knowledge Graph의 한 커뮤니티(클러스터)에 속하는 엔티티와 관계 정보야.
이 커뮤니티를 대표하는 제목(20자 이내)과 요약(3-5문장)을 생성해줘.

## 엔티티
{entities}

## 관계
{relations}

## JSON 출력 형식 (반드시 이 형식만 출력)
{{"title": "커뮤니티 제목", "summary": "커뮤니티 요약..."}}"""


class CommunityBuilder:
    """Builds community clusters from existing KG entities/relations."""

    def __init__(self, namespace: str, gemini_client=None):
        self.namespace = namespace
        self.config = get_graph_config(namespace).get('community', {})
        if gemini_client is None:
            from services.singletons import get_gemini_client
            gemini_client = get_gemini_client()
        self.gemini = gemini_client

    def build(self, skip_summary: bool = False) -> dict:
        """Full pipeline: detect communities → save → generate summaries."""
        G = self._load_kg_graph()
        min_size = self.config.get('min_community_size', 3)

        if G.number_of_nodes() < min_size:
            logger.info("[CommunityBuilder] %s: only %d nodes, skipping",
                        self.namespace, G.number_of_nodes())
            return {'communities': 0, 'summarized': 0,
                    'total_nodes': G.number_of_nodes(), 'skipped': 'insufficient_nodes'}

        communities = self._detect_communities(G)
        communities = {cid: members for cid, members in communities.items()
                       if len(members) >= min_size}

        if not communities:
            logger.info("[CommunityBuilder] %s: no communities above min_size=%d",
                        self.namespace, min_size)
            return {'communities': 0, 'summarized': 0,
                    'total_nodes': G.number_of_nodes()}

        saved = self._save_communities(communities)
        logger.info("[CommunityBuilder] %s: saved %d communities", self.namespace, saved)

        summarized = 0
        if not skip_summary:
            summarized = self._generate_summaries()
            logger.info("[CommunityBuilder] %s: summarized %d communities",
                        self.namespace, summarized)

        return {
            'communities': saved,
            'summarized': summarized,
            'total_nodes': G.number_of_nodes(),
        }

    def reset(self):
        """Delete all community data for this namespace."""
        KGCommunityMember.query.filter_by(namespace=self.namespace).delete()
        KGCommunity.query.filter_by(namespace=self.namespace).delete()
        db.session.commit()
        logger.info("[CommunityBuilder] Reset communities for namespace=%s", self.namespace)

    # ------------------------------------------------------------------
    # Step 1: Load KG as networkx graph
    # ------------------------------------------------------------------

    def _load_kg_graph(self) -> nx.Graph:
        """Load KGEntity + KGRelation into a networkx Graph."""
        G = nx.Graph()

        entities = KGEntity.query.filter_by(namespace=self.namespace).all()
        for e in entities:
            G.add_node(e.id, name=e.name, entity_type=e.entity_type,
                       description=e.description or '')

        relations = KGRelation.query.filter_by(namespace=self.namespace).all()
        for r in relations:
            if G.has_node(r.source_id) and G.has_node(r.target_id):
                G.add_edge(r.source_id, r.target_id,
                           relation_type=r.relation_type,
                           confidence=r.confidence)

        logger.info("[CommunityBuilder] Loaded graph: %d nodes, %d edges",
                    G.number_of_nodes(), G.number_of_edges())
        return G

    # ------------------------------------------------------------------
    # Step 2: Community detection
    # ------------------------------------------------------------------

    def _detect_communities(self, G: nx.Graph) -> dict[int, list[int]]:
        """Run Leiden (preferred) or Louvain (fallback) community detection."""
        resolution = self.config.get('resolution', 1.0)

        try:
            import igraph as ig
            import leidenalg

            ig_graph = ig.Graph.from_networkx(G)
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                resolution_parameter=resolution,
            )
            node_list = list(G.nodes())
            communities = {}
            for comm_id, members in enumerate(partition):
                entity_ids = [node_list[idx] for idx in members]
                communities[comm_id] = entity_ids

            logger.info("[CommunityBuilder] Leiden detected %d communities", len(communities))
            return communities

        except ImportError:
            logger.warning("[CommunityBuilder] leidenalg not installed, using Louvain fallback")
            from networkx.algorithms.community import louvain_communities
            partitions = louvain_communities(G, resolution=resolution)
            communities = {i: list(members) for i, members in enumerate(partitions)}
            logger.info("[CommunityBuilder] Louvain detected %d communities", len(communities))
            return communities

    # ------------------------------------------------------------------
    # Step 3: Save to DB
    # ------------------------------------------------------------------

    def _save_communities(self, communities: dict[int, list[int]]) -> int:
        """Persist communities to KGCommunity + KGCommunityMember."""
        self.reset()

        count = 0
        for comm_id, entity_ids in communities.items():
            community = KGCommunity(
                namespace=self.namespace,
                community_id=comm_id,
                level=0,
                member_count=len(entity_ids),
            )
            db.session.add(community)
            db.session.flush()

            for eid in entity_ids:
                db.session.add(KGCommunityMember(
                    community_id=community.id,
                    entity_id=eid,
                    namespace=self.namespace,
                ))
            count += 1

        db.session.commit()
        return count

    # ------------------------------------------------------------------
    # Step 4: LLM summary generation
    # ------------------------------------------------------------------

    def _generate_summaries(self) -> int:
        """Generate LLM summaries for communities without one."""
        communities = KGCommunity.query.filter_by(
            namespace=self.namespace, summary=None,
        ).all()

        summarized = 0
        for comm in communities:
            try:
                member_entities = (
                    db.session.query(KGEntity)
                    .join(KGCommunityMember, KGCommunityMember.entity_id == KGEntity.id)
                    .filter(KGCommunityMember.community_id == comm.id)
                    .all()
                )
                entity_ids = [e.id for e in member_entities]

                entity_text = '\n'.join(
                    f"- {e.name} ({e.entity_type}): {(e.description or '')[:200]}"
                    for e in member_entities
                )

                relations = (
                    db.session.query(KGRelation)
                    .filter(
                        KGRelation.namespace == self.namespace,
                        KGRelation.source_id.in_(entity_ids),
                        KGRelation.target_id.in_(entity_ids),
                    )
                    .all()
                )
                entity_name_map = {e.id: e.name for e in member_entities}
                relation_text = '\n'.join(
                    f"- {entity_name_map.get(r.source_id, '?')} --[{r.relation_type}]--> "
                    f"{entity_name_map.get(r.target_id, '?')}"
                    for r in relations
                )

                prompt = _SUMMARY_PROMPT.format(
                    entities=entity_text or '(없음)',
                    relations=relation_text or '(없음)',
                )

                response = self.gemini.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[{'role': 'user', 'parts': [{'text': prompt}]}],
                    config={'response_mime_type': 'application/json', 'temperature': 0.3},
                )

                text = response.text.strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1] if '\n' in text else text[3:]
                if text.endswith('```'):
                    text = text[:-3]

                result = json.loads(text)
                comm.title = result.get('title', '')[:300]
                comm.summary = result.get('summary', '')
                db.session.commit()
                summarized += 1

                logger.info("[CommunityBuilder] Summarized community %d: %s (%d members)",
                            comm.community_id, comm.title, comm.member_count)

            except Exception:
                logger.exception("[CommunityBuilder] Summary failed for community %d",
                                 comm.community_id)

        return summarized
