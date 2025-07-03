import os
import pm4py
from pm4py.objects.log.importer.xes.importer import apply as import_xes
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.heuristics import algorithm as heuristic_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer

# 로그 파일 경로 설정 (예: ./tests/input_data/example1.xes)
log_path = os.path.join("/Users/heehyeong/Downloads/practice/DomesticDeclarations.xes.gz")

# XES 로그 불러오기
log = import_xes(log_path)

# Alpha Miner 적용
net, initial_marking, final_marking = heuristic_miner.apply(log)

# 프로세스 네트워크 시각화
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)


# print(pm4py.llm.abstract_variants(log))

# for trace in log[:2]:  # 처음 두 케이스만 출력해보기
#     print("New trace")
#     for event in trace:
#         print(event)

# # 이벤트 로그를 DataFrame으로 변환
# df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

# # 1. 고유 활동명 추출 및 빈도 계산
# activity_counts = df['concept:name'].value_counts()
# unique_activities_list = activity_counts.index.tolist()

# print(f"\n총 고유 활동명 개수: {len(unique_activities_list)}")
# print("\n가장 빈번한 활동명 상위 5개:")
# print(activity_counts.head(5))

# print("\n가장 빈번하지 않은 활동명 하위 5개:")
# print(activity_counts.tail(5))