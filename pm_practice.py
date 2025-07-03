import os
from pm4py.objects.log.importer.xes.importer import apply as import_xes
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer

# 로그 파일 경로 설정 (예: ./tests/input_data/example1.xes)
log_path = os.path.join("/Users/heehyeong/Downloads/practice/Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes")

# XES 로그 불러오기
log = import_xes(log_path)

# Alpha Miner 적용
net, initial_marking, final_marking = alpha_miner.apply(log)

# 프로세스 네트워크 시각화
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)

for trace in log[:2]:  # 처음 두 케이스만 출력해보기
    print("New trace")
    for event in trace:
        print(event)

        from pm4py.objects.conversion.log import converter as log_converter

# 이벤트 로그를 DataFrame으로 변환
df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
print(df.head())