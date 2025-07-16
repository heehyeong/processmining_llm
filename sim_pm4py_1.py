import simpy
import random
import numpy as np

# --- 1. 상수 및 파라미터 정의 ---
RANDOM_SEED = 42
AVG_ARRIVAL_TIME = 10.0
NUM_EMPLOYEES = 2 # 자원 수는 일단 그대로 유지
SIM_TIME = 10000

# === 시나리오 1 파라미터 ===
# 'Add penalty' 이후, 25%의 케이스가 신용 추심으로 가지 않고 'Payment'로 향한다고 가정 (기존 0% -> 25%)
PROB_PAY_AFTER_PENALTY = 0.25 

# 시뮬레이션 결과를 저장할 변수들
results_lead_time = []
# === 측정 지표(Metric) 카운터 ===
count_credit_collection = 0
count_pay_after_penalty = 0


class Process:
    def __init__(self, env, case_id, employee_resource):
        # ... (이전 코드와 동일) ...
        self.env = env
        self.case_id = case_id
        self.employee = employee_resource
        self.arrival_time = self.env.now
        self.action = env.process(self.run_workflow())

    def run_workflow(self):
        """
        === 시나리오 1을 적용한 확률적 워크플로우 ===
        """
        global count_credit_collection, count_pay_after_penalty # 전역 카운터 사용 선언

        print(f'{self.env.now:.2f}: 케이스 {self.case_id} 도착 및 접수 시작')
        
        # 워크플로우 예시: Fine 생성 -> Penalty 추가 -> 분기
        yield self.env.process(self.execute_task('Create Fine', 5, 10))
        yield self.env.process(self.execute_task('Add penalty', 3, 5))
        
        # --- 여기가 핵심: 확률적 분기 구현 ---
        print(f'{self.env.now:.2f}: 케이스 {self.case_id}가 \'Add penalty\' 이후 경로 결정 중')
        
        if random.random() < PROB_PAY_AFTER_PENALTY:
            # [성공 경로] 25%의 확률로 'Payment'로 바로 진행 (인센티브 효과)
            count_pay_after_penalty += 1 # 카운터 증가
            print(f'{self.env.now:.2f}: 케이스 {self.case_id}가 인센티브로 Payment 경로 선택')
            yield self.env.process(self.execute_task('Payment', 10, 15))
        else:
            # [실패 경로] 나머지 75%는 기존대로 'Send for Credit Collection'으로 진행
            count_credit_collection += 1 # 카운터 증가
            print(f'{self.env.now:.2f}: 케이스 {self.case_id}가 Credit Collection 경로 선택')
            yield self.env.process(self.execute_task('Send for Credit Collection', 50, 80)) # 이 경로는 시간이 훨씬 오래 걸림

        # 모든 단계 완료 후 결과 기록
        completion_time = self.env.now
        lead_time = completion_time - self.arrival_time
        results_lead_time.append(lead_time)
        print(f'{self.env.now:.2f}: 케이스 {self.case_id} 처리 완료 (총 소요시간: {lead_time:.2f})')

    def execute_task(self, task_name, min_time, max_time):
        # ... (이전 코드와 동일) ...
        print(f'{self.env.now:.2f}: 케이스 {self.case_id}가 \'{task_name}\' 대기 시작')
        with self.employee.request() as req:
            yield req
            print(f'{self.env.now:.2f}: 케이스 {self.case_id}가 \'{task_name}\' 수행 시작')
            processing_time = random.uniform(min_time, max_time)
            yield self.env.timeout(processing_time)
            print(f'{self.env.now:.2f}: 케이스 {self.case_id}가 \'{task_name}\' 수행 완료')

# --- 2. 시뮬레이션 환경 설정 및 실행 ---
# ... (이전 코드와 동일. source 함수 등) ...

def source(env, employee_resource):
    # ... (이전 코드와 동일) ...
    case_id = 0
    while True:
        yield env.timeout(random.expovariate(1.0 / AVG_ARRIVAL_TIME))
        case_id += 1
        Process(env, case_id, employee_resource)

print('--- To-Be 시뮬레이션 (시나리오 1) 시작 ---')
random.seed(RANDOM_SEED)
env = simpy.Environment()
employee_resource = simpy.Resource(env, capacity=NUM_EMPLOYEES)
env.process(source(env, employee_resource))
env.run(until=SIM_TIME)

# --- 3. 시뮬레이션 결과 분석 ---
print('\n--- 시뮬레이션 종료 및 결과 분석 ---')
if results_lead_time:
    total_cases = len(results_lead_time)
    avg_lead_time = np.mean(results_lead_time)
    
    print(f'총 처리된 케이스 수: {total_cases}')
    print(f'평균 리드타임: {avg_lead_time:.2f} 분')
    print('\n--- 시나리오 1 관찰 지표 ---')
    print(f'"Send for Credit Collection"으로 끝난 케이스 수: {count_credit_collection} ({count_credit_collection/total_cases:.2%})')
    print(f'"Add penalty -> Payment" 경로 빈도: {count_pay_after_penalty} ({count_pay_after_penalty/total_cases:.2%})')
else:
    print('시뮬레이션 시간 동안 처리된 케이스가 없습니다.')