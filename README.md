# Applying LLMs for Process Mining Result Analysis
## Guide
### 1. Environment Setup
```
pip install -r requirements.txt
```
- revision in sax4bpm library
  - sax.core.synthesis.sax_explainability -> def getXAIPerspective(data, variants: Optional[List[str]] = None)
  - 해당 부분에 원하는 ML 모델 코드 지정
- LLM API key
  - GEMINI-2.5-flash was used here
  
 ### 2. Run the main codes
- **Baseline**: hypo_gen.py
  - generated_hypothese.txt에 결과 저장
- **SAX4BPM**: main_sax4bpm_official.py
  - results/comprehensive_analysis.txt 에 결과 저장
- **Simulation**: simulation+trial.ipynb 다운로드

---
POSTECH lab internship (AIM lab.)
