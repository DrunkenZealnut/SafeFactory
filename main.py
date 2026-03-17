#!/usr/bin/env python3
"""
Pinecone Agent CLI
Command-line interface for processing files and uploading to Pinecone.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(
        description="폴더 내 파일을 처리하여 Pinecone에 저장하는 에이전트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 폴더 처리 및 업로드
  python main.py process ./my_documents

  # 네임스페이스 지정
  python main.py process ./my_documents --namespace my-docs

  # 검색
  python main.py search "반도체 제조 공정"

  # 인덱스 통계 확인
  python main.py stats
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # Process command
    process_parser = subparsers.add_parser("process", help="폴더 처리 및 Pinecone 업로드")
    process_parser.add_argument("folder", type=str, help="처리할 폴더 경로")
    process_parser.add_argument("--namespace", "-n", type=str, default="", help="Pinecone 네임스페이스")
    process_parser.add_argument("--no-recursive", action="store_true", help="하위 폴더 미포함")
    process_parser.add_argument("--batch-size", type=int, default=50, help="배치 크기 (기본: 50)")
    process_parser.add_argument("--max-chunk-tokens", type=int, default=800, help="청크당 최대 토큰 (기본: 800)")
    process_parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small",
                               choices=["text-embedding-3-small", "text-embedding-3-large"],
                               help="임베딩 모델")
    process_parser.add_argument("--domain", type=str, default=None,
                               help="도메인 분류 (예: safety, labor)")
    process_parser.add_argument("--category", type=str, default=None,
                               help="중분류 (예: machinery, hazmat, wage, accident)")
    process_parser.add_argument("--subcategory", type=str, default=None,
                               help="소분류 (예: crane, chemical, minimum_wage, compensation)")
    process_parser.add_argument("--skip-images", action="store_true",
                               help="이미지 파일 건너뜀 (Vision API 호출 없음, 마크다운만 처리)")
    process_parser.add_argument("--force", action="store_true",
                               help="변경 여부 무시하고 모든 파일 강제 재처리")
    process_parser.add_argument("--contextual", action="store_true",
                               help="Anthropic Contextual Retrieval 활성화 (LLM 기반 청크 맥락 생성)")
    process_parser.add_argument("--context-model", type=str, default="claude-haiku-4-5-20251001",
                               help="Contextual prefix 생성 모델 (기본: claude-haiku-4-5-20251001)")

    # Search command
    search_parser = subparsers.add_parser("search", help="Pinecone에서 검색")
    search_parser.add_argument("query", type=str, help="검색 쿼리")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="결과 개수 (기본: 5)")
    search_parser.add_argument("--namespace", "-n", type=str, default="", help="Pinecone 네임스페이스")
    search_parser.add_argument("--filter-file-type", type=str, help="파일 타입 필터 (image/markdown/json)")
    search_parser.add_argument("--filter-domain", type=str, help="도메인 필터 (예: safety, labor)")
    search_parser.add_argument("--filter-category", type=str, help="중분류 필터")
    search_parser.add_argument("--filter-subcategory", type=str, help="소분류 필터")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="인덱스 통계 확인")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="벡터 삭제")
    delete_parser.add_argument("--namespace", "-n", type=str, default="", help="삭제할 네임스페이스")
    delete_parser.add_argument("--source-file", type=str, help="특정 파일의 벡터만 삭제")
    delete_parser.add_argument("--all", action="store_true", help="모든 벡터 삭제 (위험!)")

    # Build-graph command
    bg_parser = subparsers.add_parser("build-graph", help="Knowledge Graph 구축")
    bg_parser.add_argument("--namespace", "-n", type=str, required=True, help="대상 Pinecone 네임스페이스")
    bg_parser.add_argument("--batch-size", type=int, default=20, help="LLM 배치 크기 (기본: 20)")
    bg_parser.add_argument("--max-chunks", type=int, default=None, help="최대 처리 청크 수 (테스트용)")
    bg_parser.add_argument("--reset", action="store_true", help="기존 그래프 삭제 후 재구축")

    # Graph-stats command
    gs_parser = subparsers.add_parser("graph-stats", help="Knowledge Graph 통계")
    gs_parser.add_argument("--namespace", "-n", type=str, default=None, help="특정 네임스페이스 통계")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load environment variables
    load_dotenv()

    # Check required environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "document-index")

    if not openai_key:
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정하거나 환경변수로 지정해주세요.")
        sys.exit(1)

    if not pinecone_key:
        print("❌ 오류: PINECONE_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 PINECONE_API_KEY를 설정하거나 환경변수로 지정해주세요.")
        sys.exit(1)

    # Import agent after environment check
    from src.agent import PineconeAgent

    if args.command == "process":
        # Check folder exists
        if not Path(args.folder).exists():
            print(f"❌ 오류: 폴더를 찾을 수 없습니다: {args.folder}")
            sys.exit(1)

        # Initialize Contextual Retrieval if requested
        context_gen = None
        if args.contextual:
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_key:
                print("❌ 오류: --contextual 사용 시 ANTHROPIC_API_KEY 환경변수가 필요합니다.")
                print("   .env 파일에 ANTHROPIC_API_KEY를 설정해주세요.")
                sys.exit(1)
            from src.context_generator import ContextGenerator
            context_gen = ContextGenerator(
                api_key=anthropic_key,
                model=args.context_model,
            )
            print(f"🧠 Contextual Retrieval 활성화 (모델: {args.context_model})")

        print("🚀 Pinecone 에이전트 초기화 중...")
        agent = PineconeAgent(
            openai_api_key=openai_key,
            pinecone_api_key=pinecone_key,
            pinecone_index_name=index_name,
            embedding_model=args.embedding_model,
            max_chunk_tokens=args.max_chunk_tokens,
            context_generator=context_gen,
        )

        # Build extra metadata from CLI flags
        extra_metadata = {}
        if args.domain:
            extra_metadata['domain'] = args.domain
        if args.category:
            extra_metadata['category'] = args.category
        if args.subcategory:
            extra_metadata['subcategory'] = args.subcategory

        # Auto-detect NCS documents from folder path
        if 'ncs' in args.folder.lower():
            extra_metadata.setdefault('domain', 'semiconductor')
            for cat in ['반도체개발', '반도체장비', '반도체재료', '반도체제조']:
                if cat in args.folder:
                    extra_metadata.setdefault('category', cat)
                    break

        if extra_metadata:
            print(f"🏷️ 메타데이터: {extra_metadata}")

        print(f"📁 폴더 처리 시작: {args.folder}")
        result = agent.process_folder(
            folder_path=args.folder,
            namespace=args.namespace,
            recursive=not args.no_recursive,
            batch_size=args.batch_size,
            verbose=True,
            extra_metadata=extra_metadata,
            skip_images=args.skip_images,
            force=args.force
        )

        print("\n" + "="*50)
        print("📋 처리 결과 요약")
        print("="*50)
        print(f"  총 파일 수: {result.total_files}")
        print(f"  처리된 파일: {result.processed_files}")
        print(f"  생성된 청크: {result.total_chunks}")
        print(f"  업로드된 벡터: {result.uploaded_vectors}")
        print(f"  실패한 업로드: {result.failed_uploads}")

        if result.errors:
            print(f"\n⚠️ 에러 ({len(result.errors)}개):")
            for error in result.errors[:5]:
                print(f"  - {error}")

        # Print contextual retrieval cost stats
        if context_gen:
            stats = context_gen.get_stats()
            print(f"\n🧠 Contextual Retrieval 통계:")
            print(f"  LLM 호출: {stats['llm_calls']}회")
            print(f"  캐시 히트: {stats['cache_hits']}회")
            print(f"  입력 토큰: {stats['total_input_tokens']:,}")
            print(f"  출력 토큰: {stats['total_output_tokens']:,}")
            print(f"  캐시 읽기 토큰: {stats['cache_read_tokens']:,}")
            print(f"  추정 비용: ${stats['estimated_cost_usd']:.4f}")
            context_gen.close()

    elif args.command == "search":
        print("🔍 검색 중...")
        agent = PineconeAgent(
            openai_api_key=openai_key,
            pinecone_api_key=pinecone_key,
            pinecone_index_name=index_name,
            create_index_if_not_exists=False
        )

        filter_dict = {}
        if args.filter_file_type:
            filter_dict["file_type"] = args.filter_file_type
        if args.filter_domain:
            filter_dict["domain"] = args.filter_domain
        if args.filter_category:
            filter_dict["category"] = args.filter_category
        if args.filter_subcategory:
            filter_dict["subcategory"] = args.filter_subcategory
        filter_dict = filter_dict or None

        results = agent.search(
            query=args.query,
            top_k=args.top_k,
            namespace=args.namespace,
            filter=filter_dict
        )

        if not results:
            print("검색 결과가 없습니다.")
        else:
            print(f"\n🔎 검색 결과 ({len(results)}개):\n")
            for i, result in enumerate(results, 1):
                print(f"{'='*50}")
                print(f"[{i}] 유사도: {result['score']:.4f}")
                if result['metadata']:
                    print(f"    파일: {result['metadata'].get('source_file', 'N/A')}")
                    print(f"    타입: {result['metadata'].get('file_type', 'N/A')}")
                    content = result['metadata'].get('content', '')
                    if content:
                        # Truncate long content
                        if len(content) > 300:
                            content = content[:300] + "..."
                        print(f"    내용: {content}")
                print()

    elif args.command == "stats":
        print("📊 인덱스 통계 조회 중...")

        from src.pinecone_uploader import PineconeUploader
        uploader = PineconeUploader(
            api_key=pinecone_key,
            index_name=index_name,
            create_if_not_exists=False
        )

        stats = uploader.get_stats()
        print(f"\n{'='*50}")
        print(f"📈 인덱스: {index_name}")
        print(f"{'='*50}")
        print(f"  차원(Dimension): {stats['dimension']}")
        print(f"  총 벡터 수: {stats['total_vector_count']}")

        if stats['namespaces']:
            print("\n  네임스페이스별 벡터 수:")
            for ns, info in stats['namespaces'].items():
                ns_name = ns if ns else "(기본)"
                print(f"    - {ns_name}: {info.get('vector_count', 0)}개")

    elif args.command == "delete":
        if not args.all and not args.source_file:
            print("❌ 오류: --all 또는 --source-file 중 하나를 지정해야 합니다.")
            sys.exit(1)

        if args.all:
            confirm = input("⚠️ 모든 벡터를 삭제합니다. 계속하시겠습니까? (yes/no): ")
            if confirm.lower() != "yes":
                print("취소되었습니다.")
                return

        from src.pinecone_uploader import PineconeUploader
        uploader = PineconeUploader(
            api_key=pinecone_key,
            index_name=index_name,
            create_if_not_exists=False
        )

        if args.source_file:
            success = uploader.delete_by_filter(
                filter={"source_file": args.source_file},
                namespace=args.namespace
            )
            if success:
                print(f"✓ {args.source_file}의 벡터가 삭제되었습니다.")
            else:
                print("❌ 삭제 실패")
        elif args.all:
            # Delete all by deleting the namespace
            print(f"네임스페이스 '{args.namespace or '(기본)'}' 삭제 중...")
            try:
                uploader.index.delete(delete_all=True, namespace=args.namespace)
                print("✓ 모든 벡터가 삭제되었습니다.")
            except Exception as e:
                print(f"❌ 삭제 실패: {e}")


    elif args.command == "build-graph":
        # GraphRAG: build knowledge graph from Pinecone chunks
        from web_app import app
        from models import db, KGEntity, KGRelation, KGEntityChunk
        from src.graph_builder import GraphBuilder
        from pinecone import Pinecone

        with app.app_context():
            db.create_all()

            namespace = args.namespace
            print(f"🧠 Knowledge Graph 구축: namespace={namespace}")

            builder = GraphBuilder(namespace=namespace)

            if args.reset:
                print("🗑️  기존 그래프 삭제 중...")
                builder.reset()

            # Fetch chunks from Pinecone
            print("📥 Pinecone에서 청크 조회 중...")
            pc = Pinecone(api_key=pinecone_key)
            idx = pc.Index(index_name)

            # List all vector IDs in the namespace
            all_ids = []
            for id_batch in idx.list(namespace=namespace):
                all_ids.extend(id_batch)
                if args.max_chunks and len(all_ids) >= args.max_chunks:
                    all_ids = all_ids[:args.max_chunks]
                    break

            print(f"   벡터 {len(all_ids)}개 발견")

            # Fetch metadata in batches
            chunks = []
            for i in range(0, len(all_ids), 100):
                batch_ids = all_ids[i:i+100]
                fetched = idx.fetch(ids=batch_ids, namespace=namespace)
                for vid, vec in fetched.vectors.items():
                    meta = vec.metadata or {}
                    content = meta.get('content', '')
                    if content:
                        chunks.append({'id': vid, 'content': content, 'metadata': meta})

            print(f"   컨텐츠 있는 청크 {len(chunks)}개")

            if not chunks:
                print("❌ 추출 가능한 청크가 없습니다.")
                return

            # Build graph
            print(f"🔨 엔티티/관계 추출 중 (배치 크기: {args.batch_size})...")
            stats = builder.build(chunks=chunks, batch_size=args.batch_size)

            print(f"\n{'='*50}")
            print(f"📊 Knowledge Graph 구축 완료")
            print(f"{'='*50}")
            print(f"  엔티티: {stats['entities']}개")
            print(f"  관계: {stats['relations']}개")
            print(f"  엔티티-청크 매핑: {stats['entity_chunks']}개")

    elif args.command == "graph-stats":
        from web_app import app
        from models import db, KGEntity, KGRelation, KGEntityChunk
        from sqlalchemy import func

        with app.app_context():
            if args.namespace:
                namespaces = [args.namespace]
            else:
                namespaces = [
                    r[0] for r in
                    db.session.query(KGEntity.namespace).distinct().all()
                ]

            if not namespaces:
                print("Knowledge Graph가 비어 있습니다.")
                return

            print(f"\n{'='*50}")
            print(f"📊 Knowledge Graph 통계")
            print(f"{'='*50}")
            for ns in namespaces:
                e_count = db.session.query(KGEntity).filter_by(namespace=ns).count()
                r_count = db.session.query(KGRelation).filter_by(namespace=ns).count()
                ec_count = db.session.query(KGEntityChunk).filter_by(namespace=ns).count()
                print(f"\n  [{ns}]")
                print(f"    엔티티: {e_count}개")
                print(f"    관계: {r_count}개")
                print(f"    엔티티-청크 매핑: {ec_count}개")

                # Top entity types
                type_counts = (
                    db.session.query(KGEntity.entity_type, func.count(KGEntity.id))
                    .filter_by(namespace=ns)
                    .group_by(KGEntity.entity_type)
                    .all()
                )
                if type_counts:
                    print(f"    엔티티 타입:")
                    for etype, cnt in type_counts:
                        print(f"      {etype}: {cnt}개")


if __name__ == "__main__":
    main()
