from sax.core.synthesis.llms.base_llm import ModelTypes

SAX4BPM_CONFIG = {
    # 데이터 설정
    'data_file': "dataset/Road_Traffic_Fine_Management_Process.xes",
    
    'model_type': ModelTypes.GEMINI,  # OpenAI → Gemini로 변경
    'model_name': "gemini-2.5-flash",
    'temperature': 0.7,
    
    # 분석 설정
    'comprehensive_query': """
    Based on the comprehensive multi-perspective analysis of this business process, 
    please provide:
    1. Three specific optimization hypotheses with supporting data
    2. Root cause analysis of performance bottlenecks
    3. Actionable recommendations with expected measurable impact
    4. Simulation scenarios for testing improvements
    """,
    
    # 출력 설정
    'output_directory': './results/',
    'save_individual_files': True,
    'save_json_summary': True
}
