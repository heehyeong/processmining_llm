if __name__ == "__main__":

    import os
    import pm4py.llm
    from pm4py.objects.log.importer.xes.importer import apply as import_xes
    from pm4py.algo.discovery.heuristics import algorithm as heuristic_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer

    # 로그 파일 경로 설정 (예: ./tests/input_data/example1.xes)
    log_path = os.path.join("dataset/Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz")

    # XES 로그 불러오기
    log = import_xes(log_path)

    # Alpha Miner 적용
    net, initial_marking, final_marking = heuristic_miner.apply(log)

    # # 프로세스 네트워크 시각화
    # gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    # pn_visualizer.view(gviz)

    pm_variants = pm4py.llm.abstract_variants(log)
    pm_model = pm4py.llm.abstract_dfg(log)
    pm_log = pm4py.llm.abstract_log_features(log)
    user_query = '''
QUESTION: Based on the detailed analysis, can you formulate specific, data-driven hypotheses for process optimization?

Please provide detailed, data-driven hypotheses based on the comprehensive analysis above. Consider:
- Process patterns and statistics from the process view
- Concrete, actionable recommendations for process improvement

Format your response with:
1. Three specific hypotheses with supporting data
2. Simulation scenarios for each hypothesis
3. Expected outcomes with measurable KPIs
4. Implementation considerations
'''

    pm_text = f"""{pm_variants}
    process model: {pm_model}
    event log: {pm_log}
    user query: {user_query}"""

    # LLM 프롬프트 저장
    with open('pm_text.txt', 'w', encoding='utf-8') as f:
        f.write(pm_text)