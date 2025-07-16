import os
from google import genai
from google.genai import types  # 이 import가 핵심!
import json
from datetime import datetime

def safe_file_read(file_path):
    """안전한 파일 읽기 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def generate_hypotheses_with_gemini(integrated_prompt, api_key, model="gemini-2.5-flash"):
    """
    수정된 Gemini API 호출 함수
    """
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=integrated_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=4000,
                top_p=0.9
            )
        )
        
        print("API 호출 성공!")
        
        return {
            "success": True,
            "response": response.text,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    # 파일 읽기
    pm_text = safe_file_read('results/pm_text.txt')
    query = '''
        Based on the textual abstraction of this business process, 
        please provide:
        1. Three specific optimization hypotheses with supporting data
        2. Root cause analysis of performance bottlenecks
        3. Actionable recommendations with expected measurable impact
        4. Simulation scenarios for testing improvements
    '''
    # synthesized_explanation = safe_file_read('results/enhanced_sax4bpm_prompt.txt')
    
    if pm_text is None:
        print("❌ 필요한 파일을 읽을 수 없습니다.")
        return
    
    # 프롬프트 생성
    total_prompt = f'''Human: {query}
                        process abstraction: {pm_text}'''
#   additional explanation: {synthesized_explanation}'''
    
    # total_prompt = f'''process abstraction: {pm_text}'''
    
    print(f"프롬프트 생성 완료 (길이: {len(total_prompt)} 문자)")
    
    # API 호출
    api_key = 'your-key'
    result = generate_hypotheses_with_gemini(total_prompt, api_key)
    
    # 결과 처리
    if result["success"]:
        print("가설 생성 성공!")
        print("\n--- 생성된 가설 ---")
        print(result["response"])
        
        # 파일로 저장
        with open('generated_hypotheses.txt', 'w', encoding='utf-8') as f:
            f.write(result["response"])
        print("\n📁 결과가 'generated_hypotheses.txt'에 저장되었습니다.")
        
    else:
        print(f"가설 생성 실패: {result['error']}")

if __name__ == "__main__":
    main()