import warnings
import pandas as pd
import pm4py
import os
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import sax
from sax.core.process_data.raw_event_data import RawEventData
import sax.core.causal_process_discovery.causal_discovery as cd
import sax.core.process_mining.process_mining as pm

try:
    print("✅ 'sax' 라이브러리 모듈을 성공적으로 임포트했습니다.")
except ImportError:
    print("❌ 'sax' 라이브러리 모듈을 찾을 수 없습니다.")
    print("   프로젝트에 'sax' 폴더 구조가 올바르게 포함되어 있는지 확인해주세요.")
    cd = None

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 이벤트 로그 파일 임포트
fileName = "Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz"
df = None

if not os.path.exists(fileName):
    print(f"❌ 오류: 파일 '{fileName}'을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
else:
    df = pm.import_xes(fileName)
    print("✅ 이벤트 로그를 성공적으로 불러왔습니다.")

# sax 모듈과 로그 파일이 모두 준비된 경우에만 분석 진행
if cd and df is not None:
    try:
        # --- Part 1: Causal Discovery using the provided SAX module ---
        print("\n--- 1단계: SAX 모듈을 이용한 인과관계 분석 시작 ---")

        # 1.1. SAX 모듈이 요구하는 RawEventData 객체 생성
        # (실제 RawEventData 클래스의 생성 방식에 따라 조정이 필요할 수 있습니다)
        # data_object = RawEventData(df) 
        # print("\n✅ SAX 라이브러리용 RawEventData 객체 생성 완료.")

        # 1.2. 제공된 모듈의 함수를 호출하여 인과관계 분석 수행
        # 이 함수는 내부에 변형(variant) 분석, 통합, 게이트웨이 발견 등 복잡한 로직을 포함합니다.
        causal_result = cd.discover_causal_dependencies(df)
        print("\n✅ SAX 모듈을 통해 인과 모델을 성공적으로 발견했습니다.")

        # 1.3. 분석 결과를 Graphviz 객체로 변환하여 시각화 및 저장
        causal_graph = cd.view_causal_dependencies(causal_result)
        
        try:
            graph_output_filename = 'causal_graph_with_sax_module'
            causal_graph.render(graph_output_filename, format='png', view=False, cleanup=True)
            print(f"\n✅ 인과 관계 그래프를 '{graph_output_filename}.png' 파일로 저장했습니다.")
        except Exception as e:
            print(f"\n❌ 인과 그래프 저장 중 오류: {e}")
            print("   Graphviz가 시스템에 설치되어 있고 PATH가 설정되었는지 확인해주세요.")

        # --- Part 2: XAI Analysis (SHAP) ---
        print("\n\n--- 2단계: XAI를 이용한 피처 중요도 분석 시작 ---")
        
        # 2.1. XAI 분석을 위한 데이터 준비 (인과 분석과 별개로 진행)
        # 활동 발생 유무(0/1) 행렬 생성
        df = df.getData()
        binary_activity_matrix = (pd.crosstab(df['case:concept:name'], df['concept:name']) > 0).astype(int)
        
        # 예측 목표(Target) 설정: 케이스별 총 소요 시간 (초)
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        case_durations = df.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
        case_durations['duration_seconds'] = (case_durations['max'] - case_durations['min']).dt.total_seconds()
        
        # 피처(X)와 타겟(y) 데이터 정렬
        X = binary_activity_matrix
        y = case_durations.loc[X.index]['duration_seconds']
        print("\n✅ 예측 목표(케이스 소요 시간) 및 피처 데이터 준비 완료.")

        # 2.2. 머신러닝 모델(XGBoost) 학습
        ml_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        ml_model.fit(X, y)
        print("\n✅ 케이스 소요 시간 예측을 위한 XGBoost 모델 학습 완료.")

        # 2.3. SHAP을 이용한 XAI 분석
        explainer = shap.TreeExplainer(ml_model)
        shap_values = explainer.shap_values(X)
        print("\n✅ SHAP 분석을 통해 피처(활동) 중요도 계산 완료.")

        # 2.4. XAI 결과 시각화 및 저장
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title('XAI Feature Importance for Case Duration')
        plt.xlabel('SHAP Value (Impact on Case Duration)')
        xai_graph_filename = 'xai_feature_importance.png'
        plt.savefig(xai_graph_filename, bbox_inches='tight')
        plt.close()
        print(f"\n✅ XAI 피처 중요도 그래프를 '{xai_graph_filename}' 파일로 저장했습니다.")

        # 2.5. XAI 분석 결과를 LLM을 위한 텍스트로 변환
        def generate_xai_summary(shap_values, features):
            mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=features.columns)
            sorted_features = mean_abs_shap.sort_values(ascending=False)
            
            summary_lines = ["--- XAI 분석 결과 요약 (케이스 소요 시간에 대한 영향력) ---"]
            for feature_name, importance in sorted_features.items():
                summary_lines.append(f"- '{feature_name}' 활동은 케이스 소요 시간에 영향을 미치는 주요 요인입니다 (중요도: {importance:.4f}).")
            summary_lines.append("----------------------------------------------------")
            return "\n".join(summary_lines)

        xai_summary_text = generate_xai_summary(shap_values, X)
        print("\n--- LLM 전송용 XAI 텍스트 요약 ---")
        print(xai_summary_text)

    except Exception as e:
        print(f"\n❌ 분석 과정 중 예상치 못한 오류 발생: {e}")


# --- Part 3: Process Mining for Process View ---
print("\n\n--- 3단계: 프로세스 마이닝을 통한 프로세스 뷰 생성 ---")

# 3.1. Process discovery using PM4Py
process_model = pm4py.discover_petri_net_inductive(df)
print("✅ 프로세스 모델 발견 완료.")

# 3.2. Generate process view summary for LLM
def generate_process_summary(df):
    activities = df['concept:name'].unique()
    activity_counts = df['concept:name'].value_counts()
    
    summary_lines = ["--- 프로세스 뷰 요약 ---"]
    summary_lines.append(f"총 활동 수: {len(activities)}")
    summary_lines.append("주요 활동별 발생 빈도:")
    
    for activity, count in activity_counts.head(10).items():
        summary_lines.append(f"- {activity}: {count}회")
    
    summary_lines.append("----------------------------------------------------")
    return "\n".join(summary_lines)

process_summary_text = generate_process_summary(df)
print("\n--- LLM 전송용 프로세스 뷰 텍스트 요약 ---")
print(process_summary_text)

# --- Part 4: Knowledge Synthesis for LLM (SAX4BPM Core) ---
print("\n\n--- 4단계: SAX4BPM 지식 통합 및 LLM 프롬프트 생성 ---")

def synthesize_knowledge_for_llm(process_summary, causal_summary, xai_summary, user_query):
    """
    SAX4BPM 프레임워크의 NLP4X 서비스 구현
    다양한 지식 요소들을 LLM 입력용으로 통합
    """
    
    prompt_template = """
PROCESS VIEW: The following represents the business process structure and activity patterns:
{process_view}

CAUSAL VIEW: The following represents causal execution dependencies among process activities:
{causal_view}

XAI VIEW: The following represents feature importance analysis for process outcomes:
{xai_view}

The above text includes three perspectives about a business process, consisting of a process view, a causal view, and an XAI view.

QUESTION: Can you give the briefest and most concise explanation to "{user_query}", considering the views above: [process view], [causal view], and [XAI view]?
"""
    
    return prompt_template.format(
        process_view=process_summary,
        causal_view=causal_summary,
        xai_view=xai_summary,
        user_query=user_query
    )

# 4.1. Generate causal summary from causal_result
def generate_causal_summary(causal_result):
    """인과 분석 결과를 LLM용 텍스트로 변환"""
    summary_lines = ["--- 인과 관계 분석 결과 요약 ---"]
    
    # causal_result 구조에 따라 조정 필요
    if hasattr(causal_result, 'causal_matrix') or isinstance(causal_result, dict):
        summary_lines.append("발견된 주요 인과 관계:")
        # 실제 causal_result 구조에 맞게 수정
        summary_lines.append("- 활동 간 인과적 실행 종속성이 발견되었습니다.")
    else:
        summary_lines.append("인과 관계 분석이 완료되었습니다.")
    
    summary_lines.append("----------------------------------------------------")
    return "\n".join(summary_lines)

causal_summary_text = generate_causal_summary(causal_result)

# 4.2. Create synthesized prompt
user_question = "도로 교통 벌금 관리 프로세스에서 처리 지연이 발생하는 이유는 무엇인가요?"
synthesized_prompt = synthesize_knowledge_for_llm(
    process_summary_text,
    causal_summary_text, 
    xai_summary_text,
    user_question
)

print("\n--- SAX4BPM 통합 프롬프트 (LLM 입력용) ---")
print(synthesized_prompt)

# 4.3. Save prompt to file for LLM usage
with open('sax4bpm_synthesized_prompt.txt', 'w', encoding='utf-8') as f:
    f.write(synthesized_prompt)
print("\n✅ 통합 프롬프트를 'sax4bpm_synthesized_prompt.txt' 파일로 저장했습니다.")
