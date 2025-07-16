import pm4py

# 1. 이벤트 로그 불러오기
log = pm4py.read_xes('dataset/Road_Traffic_Fine_Management_Process.xes') # 데이터프레임 출력

# 2. 케이스별 소요 시간(리드타임) 계산
case_durations = pm4py.get_all_case_durations(log)

# 3. 평균 리드타임 계산 (분 단위로 변환)
average_lead_time_seconds = sum(case_durations) / len(case_durations)
average_lead_time_minutes = average_lead_time_seconds / 60

print(f"총 케이스 수: {len(log)}")
print(f"평균 리드타임: {average_lead_time_minutes:.2f} 분")