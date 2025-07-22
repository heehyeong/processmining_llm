import os
import json
import pm4py
import langchain
from datetime import datetime
from sax.core.process_data.raw_event_data import RawEventData
from sax.core.synthesis.llms.base_llm import BaseLLM, ModelTypes
from sax.core.causal_process_discovery.causal_constants import Modality
import sax.core.process_mining.process_mining as pm
from sax.core.synthesis.sax_explainability import (
    getSyntethis, getModel, getExplanations, 
    createDocumentContextRetriever
)
from config import SAX4BPM_CONFIG
from utils import save_results, create_output_directory

langchain.debug = True

# os.environ['OPENAI_KEY'] = 'YOUR KEY'
os.environ['GOOGLE_API_KEY'] = "YOUR KEY"

def main():
    """메인 실행 함수"""
    
    print("=== SAX4BPM 공식 모듈 분석 파이프라인 ===")
    
    # 결과 저장 디렉토리 생성
    create_output_directory()
    
    # 1. 데이터 로드 및 모델 설정
    raw_data, model = setup_data_and_model()
    
    # 2. 종합 분석
    comprehensive_result = run_comprehensive_analysis(raw_data, model)
    
    # # 3. 개별 관점 분석 (상세 분석용)
    # individual_results = run_individual_perspective_analysis(raw_data, model)
    
    # 4. 결과 저장 및 출력
    all_results = save_all_results(comprehensive_result)

    return all_results

def setup_data_and_model():
    """데이터 로드 및 모델 설정"""
    
    event_log = pm.import_xes(SAX4BPM_CONFIG['data_file'])
    df = event_log.getData()
    
    mandatory_properties = event_log.getMandatoryProperties()
    optional_properties = event_log.getOptionalProperties()
    
    raw_data = RawEventData(
        data=df,
        mandatory_properties=mandatory_properties,
        optional_properties=optional_properties
    )
    print("RawEventData 객체 생성 완료 (PM4Py)")
        

    # 모델 설정
    model = getModel(
        modelType=SAX4BPM_CONFIG['model_type'],
        modelName=SAX4BPM_CONFIG['model_name'],
        temperature=SAX4BPM_CONFIG['temperature']
    )
    print("LLM 모델 설정 완료")
    
    return raw_data, model


def run_comprehensive_analysis(raw_data, model):
    print("\n--- 종합 SAX4BPM 분석 시작 ---")
    
    # 기존: 수동으로 각 단계 실행 + 프롬프트 조합
    # 새로운: 공식 모듈의 getSyntethis로 한 번에 처리
    comprehensive_result = getSyntethis(
        data=raw_data,
        query=SAX4BPM_CONFIG['comprehensive_query'],
        model=model,
        causal=True,      # causal discovery
        process=True,     # process discovery
        xai=True,         # xai
        rag=False,
        modality=Modality.CHAIN, #PARENT는 오류 발생
        prior_knowledge=True,
        p_value_threshold=0.05
    )
    
    print("종합 분석 완료")
    return comprehensive_result

def run_individual_perspective_analysis(raw_data, model):
    """개별 관점별 분석 (기존 코드의 상세 분석 대체)"""
    
    print("\n--- 개별 관점 분석 시작 ---")
    
    base_query = "What are the key performance factors in this process?"
    
    # 공식 모듈로 각 관점별 분석
    results = {}
    
    # 프로세스 관점만
    results['process_only'] = getSyntethis(
        data=raw_data, query=base_query, model=model,
        causal=False, process=True, xai=False, rag=False
    )
    
    # 인과 관점만
    results['causal_only'] = getSyntethis(
        data=raw_data, query=base_query, model=model,
        causal=True, process=False, xai=False, rag=False,
        modality=Modality.CHAIN, prior_knowledge=True
    )
    
    # XAI 관점만
    results['xai_only'] = getSyntethis(
        data=raw_data, query=base_query, model=model,
        causal=False, process=False, xai=True, rag=False
    )
    
    print("개별 관점 분석 완료")
    return results

def save_all_results(comprehensive_result):
    """모든 결과 저장 (기존 코드의 Part 4 대체)"""
 
    # 새로운: 구조화된 결과 저장
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "comprehensive_analysis": comprehensive_result,
        # "individual_perspectives": individual_results
    }
    
    save_results(all_results)
    print("모든 결과 저장 완료")

if __name__ == "__main__":
    main()
