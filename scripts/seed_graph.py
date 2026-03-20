"""Seed the default graph data if empty."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_app import app
from models import db, GraphNode, GraphEdge, SystemSetting

NODES = [
    ('w','웨이퍼 제조','단결정 실리콘 잉곳→웨이퍼 가공','m','#cba6f7',24,1,None),
    ('ox','산화 공정','SiO₂ 절연막 형성','m','#cba6f7',24,2,None),
    ('ph','포토리소그래피','마스크 패턴을 웨이퍼에 전사','m','#cba6f7',24,3,None),
    ('et','식각 공정','불필요 박막 선택적 제거','m','#cba6f7',24,4,None),
    ('dep','박막 증착','CVD/PVD/ALD 등 박막 형성','m','#cba6f7',24,5,None),
    ('met','금속 배선','다층 Cu/Al 배선 형성','m','#cba6f7',24,6,None),
    ('eds','EDS 검사','전기적 다이 분류','m','#cba6f7',24,7,None),
    ('pkg','패키징','칩 보호·방열·외부 연결','m','#cba6f7',24,8,None),
    ('ingot','잉곳 성장','고순도 실리콘 단결정 성장','s','#89b4fa',14,None,'w'),
    ('slice','슬라이싱','잉곳을 얇은 웨이퍼로 절단','s','#89b4fa',14,None,'w'),
    ('polish','연마/세정','표면 평탄화 및 불순물 제거','s','#89b4fa',14,None,'w'),
    ('cz','CZ법','초크랄스키법','d','#f9e2af',9,None,'ingot'),
    ('fz','FZ법','부유대법','d','#f9e2af',9,None,'ingot'),
    ('dry_ox','건식 산화','O₂ 가스 산화막','s','#89b4fa',14,None,'ox'),
    ('wet_ox','습식 산화','H₂O 증기 산화막','s','#89b4fa',14,None,'ox'),
    ('pr','감광제 도포','포토레지스트 스핀코팅','s','#89b4fa',14,None,'ph'),
    ('exp','노광','Stepper/Scanner','s','#89b4fa',14,None,'ph'),
    ('dev','현상','감광제 선택적 용해','s','#89b4fa',14,None,'ph'),
    ('euv','EUV','극자외선 13.5nm','d','#f9e2af',9,None,'exp'),
    ('duv','DUV/ArF','심자외선 193nm','d','#f9e2af',9,None,'exp'),
    ('dry_et','건식 식각','플라즈마 이방성 식각','s','#89b4fa',14,None,'et'),
    ('wet_et','습식 식각','화학 용액 등방성 식각','s','#89b4fa',14,None,'et'),
    ('rie','RIE','반응성 이온 식각','d','#f9e2af',9,None,'dry_et'),
    ('icp','ICP','유도결합 플라즈마','d','#f9e2af',9,None,'dry_et'),
    ('cvd','CVD','화학 기상 증착','s','#89b4fa',14,None,'dep'),
    ('pvd','PVD','물리 기상 증착','s','#89b4fa',14,None,'dep'),
    ('ald','ALD','원자층 증착','s','#89b4fa',14,None,'dep'),
    ('pecvd','PECVD','플라즈마 보조 CVD','d','#f9e2af',9,None,'cvd'),
    ('lpcvd','LPCVD','저압 CVD','d','#f9e2af',9,None,'cvd'),
    ('sput','스퍼터링','Ar 이온 타겟 충돌','d','#f9e2af',9,None,'pvd'),
    ('dam','다마신 공정','트렌치 Cu 매립','s','#89b4fa',14,None,'met'),
    ('bar','배리어 메탈','Cu 확산 방지층','s','#89b4fa',14,None,'met'),
    ('ddam','듀얼 다마신','비아+트렌치 동시','d','#f9e2af',9,None,'dam'),
    ('ecp','전해도금','Cu 전기도금','d','#f9e2af',9,None,'dam'),
    ('wtest','웨이퍼 테스트','프로브 카드 검사','s','#89b4fa',14,None,'eds'),
    ('burn','번인 테스트','가속 수명 시험','s','#89b4fa',14,None,'eds'),
    ('dice','다이싱','웨이퍼→개별 칩','s','#89b4fa',14,None,'pkg'),
    ('bond','본딩','전기 연결','s','#89b4fa',14,None,'pkg'),
    ('mold','몰딩','에폭시 밀봉','s','#89b4fa',14,None,'pkg'),
    ('wb','와이어 본딩','금/구리선 연결','d','#f9e2af',9,None,'bond'),
    ('fc','플립칩','범프 직접 접합','d','#f9e2af',9,None,'bond'),
    ('tsv','TSV','실리콘 관통 비아','d','#f9e2af',9,None,'bond'),
    ('cmp','CMP','화학적 기계적 연마','sh','#a6e3a1',15,None,None),
    ('plasma','플라즈마 기술','식각·증착 핵심 기반','sh','#a6e3a1',15,None,None),
    ('clean','세정 공정','웨이퍼 표면 정화','sh','#a6e3a1',15,None,None),
    ('ion','이온 주입','도핑 전기적 특성 제어','sh','#a6e3a1',15,None,None),
]

FLOW = [('w','ox'),('ox','ph'),('ph','et'),('et','dep'),('dep','met'),('met','eds'),('eds','pkg')]
SHARED = [('w','cmp'),('dep','cmp'),('met','cmp'),('et','plasma'),('dep','plasma'),
          ('w','clean'),('ph','clean'),('et','clean'),('ox','ion'),('et','ion')]

with app.app_context():
    db.create_all()
    if GraphNode.query.filter_by(namespace='default').count() > 0:
        print(f'Graph already exists ({GraphNode.query.filter_by(namespace="default").count()} nodes). Skipping.')
        sys.exit(0)

    for n in NODES:
        db.session.add(GraphNode(
            node_id=n[0], label=n[1], description=n[2], node_type=n[3],
            color=n[4], radius=n[5], order_num=n[6], parent_node_id=n[7], namespace='default'))
    for e in FLOW:
        db.session.add(GraphEdge(source_id=e[0], target_id=e[1], is_flow=True, namespace='default'))
    for e in SHARED:
        db.session.add(GraphEdge(source_id=e[0], target_id=e[1], is_flow=False, namespace='default'))

    # Set display name
    if not SystemSetting.query.filter_by(key='graph_name:default').first():
        db.session.add(SystemSetting(key='graph_name:default', value='반도체 8대 공정'))

    db.session.commit()
    print(f'Seeded {GraphNode.query.filter_by(namespace="default").count()} nodes, '
          f'{GraphEdge.query.filter_by(namespace="default").count()} edges')
