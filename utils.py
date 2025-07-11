"""
SAX4BPM 분석 유틸리티 함수들
"""

import os
import json
from datetime import datetime
from config import SAX4BPM_CONFIG

def create_output_directory():
    """결과 저장 디렉토리 생성"""
    os.makedirs(SAX4BPM_CONFIG['output_directory'], exist_ok=True)

def save_results(all_results):
    """결과 저장 함수"""
    
    output_dir = SAX4BPM_CONFIG['output_directory']
    
    # 1. 종합 분석 결과 텍스트 저장
    with open(f"{output_dir}comprehensive_analysis.txt", 'w', encoding='utf-8') as f:
        f.write("=== SAX4BPM integrated analysis ===\n\n")
        f.write(all_results['comprehensive_analysis'])
    
    # # 2. 개별 관점 결과 저장
    # if SAX4BPM_CONFIG['save_individual_files']:
    #     for perspective, result in all_results['individual_perspectives'].items():
    #         with open(f"{output_dir}{perspective}_analysis.txt", 'w', encoding='utf-8') as f:
    #             f.write(f"=== {perspective.upper()} analysis result ===\n\n")
    #             f.write(result)
    
    # # 3. 불일치 분석 결과 저장
    # with open(f"{output_dir}discrepancies_analysis.txt", 'w', encoding='utf-8') as f:
    #     f.write("=== Process-Causal inequality analysis ===\n\n")
    #     for i, disc in enumerate(all_results['process_causal_discrepancies'], 1):
    #         f.write(f"{i}. {disc}\n")
    
    # # 4. 전체 결과 JSON 저장
    # if SAX4BPM_CONFIG['save_json_summary']:
    #     with open(f"{output_dir}complete_sax4bpm_results.json", 'w', encoding='utf-8') as f:
    #         json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

def print_summary(results):
    """결과 요약 출력"""
    print("\n=== 생성된 파일 ===")
    print("- comprehensive_analysis.txt: 종합 분석 결과")
    # print("- process_only_analysis.txt: 프로세스 관점 분석")
    # print("- causal_only_analysis.txt: 인과 관점 분석") 
    # print("- xai_only_analysis.txt: XAI 관점 분석")
    # print("- discrepancies_analysis.txt: 불일치 분석")
    # print("- complete_sax4bpm_results.json: 전체 결과")
