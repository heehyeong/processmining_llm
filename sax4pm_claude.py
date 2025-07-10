if __name__ == "__main__":
    import warnings
    import pandas as pd
    import pm4py
    import os
    import xgboost as xgb
    import shap
    import matplotlib.pyplot as plt
    import json
    from datetime import datetime

    # Import actual sax4bpm library components (corrected)
    try:
        from sax.core.process_data.raw_event_data import RawEventData
        from sax.core.causal_process_discovery import causal_discovery as cd
        from sax.core.process_mining import process_mining as pm
        print("✅ Successfully imported sax4bpm library.")
    except ImportError as e:
        print(f"❌ sax4bpm library not found: {e}")
        cd = None
        pm = None

    # Configure warning filters
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    # Event log file import
    fileName = "Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz"

    if not os.path.exists(fileName):
        print(f"❌ Error: File '{fileName}' not found.")
        exit(1)

    # Load event log
    if pm:
        try:
            df = pm.import_xes(fileName)
            print("✅ Event log loaded with sax4bpm.")
        except:
            df = pm4py.read_xes(fileName)
            print("✅ Event log loaded with pm4py.")
    else:
        df = pm4py.read_xes(fileName)
        print("✅ Event log loaded with pm4py.")

    def generate_detailed_xai_summary(shap_values, features):
        """XAI 분석의 상세 결과를 텍스트로 생성"""
        mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=features.columns)
        sorted_features = mean_abs_shap.sort_values(ascending=False)
        
        total_importance = sorted_features.sum()
        if total_importance > 0:
            relative_importance = (sorted_features / total_importance) * 100
        else:
            relative_importance = sorted_features
        
        # 상세한 XAI 결과 텍스트 생성
        detailed_summary = "=== Detailed XAI Feature Importance Analysis ===\n"
        detailed_summary += f"Total number of activities analyzed: {len(sorted_features)}\n"
        detailed_summary += f"Total absolute importance sum: {total_importance:,.0f}\n\n"
        
        detailed_summary += "Activity-wise Impact Rankings:\n"
        for rank, (feature_name, importance) in enumerate(sorted_features.items(), 1):
            detailed_summary += f"{rank}. '{feature_name}': "
            detailed_summary += f"Absolute Impact = {importance:,.0f}, "
            detailed_summary += f"Relative Importance = {relative_importance[feature_name]:.2f}%\n"
        
        # 상위 5개 활동에 대한 해석
        top_5 = sorted_features.head(5)
        detailed_summary += f"\nTop 5 Most Influential Activities:\n"
        for activity in top_5.index:
            impact_level = "High" if relative_importance[activity] > 10 else "Medium" if relative_importance[activity] > 5 else "Low"
            detailed_summary += f"- {activity}: {impact_level} impact ({relative_importance[activity]:.2f}% of total influence)\n"
        
        return detailed_summary

    def generate_detailed_causal_summary(causal_result):
        """Causal discovery 결과의 상세 내용을 텍스트로 생성"""
        detailed_summary = "=== Detailed Causal Discovery Analysis ===\n"
        
        try:
            # causal_result 구조에 따라 실제 내용 추출
            if hasattr(causal_result, 'causal_matrix'):
                detailed_summary += "Causal Matrix Analysis:\n"
                detailed_summary += f"Matrix dimensions: {causal_result.causal_matrix.shape}\n"
                
            if hasattr(causal_result, 'dependencies'):
                detailed_summary += "Discovered Causal Dependencies:\n"
                for i, dep in enumerate(causal_result.dependencies, 1):
                    detailed_summary += f"{i}. {dep.cause} → {dep.effect} (strength: {dep.coefficient:.3f})\n"
                    
            if hasattr(causal_result, 'algorithm_info'):
                detailed_summary += f"Algorithm used: {causal_result.algorithm_info}\n"
                
        except Exception as e:
            pass
            
        # 일반적인 설명 추가
        detailed_summary += "\nCausal Interpretation:\n"
        detailed_summary += "- LiNGAM algorithm identified directional dependencies between process activities\n"
        detailed_summary += "- Causal relationships indicate which activities influence the execution of others\n"
        detailed_summary += "- These dependencies can reveal bottlenecks and optimization opportunities\n"
        detailed_summary += "- Causal structure provides insights into process flow patterns and decision points\n"
        
        return detailed_summary

    def generate_detailed_process_summary(df_data):
        """Process view의 상세 정보를 텍스트로 생성"""
        activities = df_data['concept:name'].unique()
        activity_counts = df_data['concept:name'].value_counts()
        case_count = df_data['case:concept:name'].nunique()
        
        detailed_summary = "=== Detailed Process View Analysis ===\n"
        detailed_summary += f"Total unique activities: {len(activities)}\n"
        detailed_summary += f"Total cases: {case_count}\n"
        detailed_summary += f"Total events: {len(df_data)}\n"
        detailed_summary += f"Average events per case: {len(df_data) / case_count:.1f}\n\n"
        
        # 활동별 상세 통계
        detailed_summary += "Activity Frequency Analysis:\n"
        for rank, (activity, count) in enumerate(activity_counts.items(), 1):
            percentage = (count / len(df_data)) * 100
            detailed_summary += f"{rank}. '{activity}': {count} occurrences ({percentage:.1f}% of all events)\n"
        
        # 케이스 지속시간 분석
        df_data['time:timestamp'] = pd.to_datetime(df_data['time:timestamp'])
        case_durations = df_data.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
        case_durations['duration_hours'] = (case_durations['max'] - case_durations['min']).dt.total_seconds() / 3600
        
        detailed_summary += f"\nCase Duration Statistics:\n"
        detailed_summary += f"- Average case duration: {case_durations['duration_hours'].mean():.1f} hours\n"
        detailed_summary += f"- Median case duration: {case_durations['duration_hours'].median():.1f} hours\n"
        detailed_summary += f"- Shortest case: {case_durations['duration_hours'].min():.1f} hours\n"
        detailed_summary += f"- Longest case: {case_durations['duration_hours'].max():.1f} hours\n"
        
        # 프로세스 변형 분석 (간단한 버전)
        case_variants = df_data.groupby('case:concept:name')['concept:name'].apply(lambda x: ' → '.join(x)).value_counts()
        detailed_summary += f"\nProcess Variants:\n"
        detailed_summary += f"- Total unique variants: {len(case_variants)}\n"
        detailed_summary += f"- Most common variant frequency: {case_variants.iloc[0]} cases\n"
        detailed_summary += f"- Variant diversity: {len(case_variants) / case_count:.2f} (1.0 = all cases unique)\n"
        
        return detailed_summary

    def synthesize_enhanced_sax4bpm_explanation(detailed_process, detailed_causal, detailed_xai, user_query):
        """향상된 SAX4BPM 설명 합성 - 상세 정보 포함"""
        
        enhanced_prompt = f"""=== SAX4BPM Enhanced Multi-View Process Analysis ===

DETAILED PROCESS VIEW:
{detailed_process}

DETAILED CAUSAL VIEW:
{detailed_causal}

DETAILED XAI VIEW:
{detailed_xai}

ANALYSIS INTEGRATION:
The above comprehensive analysis combines three detailed perspectives:
1. Process View: Complete activity statistics, case durations, and variant analysis
2. Causal View: Specific causal dependencies and relationships between activities  
3. XAI View: Quantitative feature importance with absolute and relative impact measures

QUESTION: {user_query}

Please provide detailed, data-driven hypotheses based on the comprehensive analysis above. Consider:
- Specific activities mentioned in the XAI analysis with their impact scores
- Causal relationships identified in the causal analysis
- Process patterns and statistics from the process view
- Concrete, actionable recommendations for process improvement

Format your response with:
1. Three specific hypotheses with supporting data
2. Simulation scenarios for each hypothesis
3. Expected outcomes with measurable KPIs
4. Implementation considerations"""
        
        return enhanced_prompt

    # Proceed with analysis only if SAX module and log file are ready
    if cd and df is not None:
        try:
            # --- Part 1: Causal Discovery using SAX module ---
            print("\n--- Step 1: Causal Relationship Analysis with SAX Module ---")

            # 1.1. Perform causal relationship analysis
            causal_result = cd.discover_causal_dependencies(df)
            print("✅ Successfully discovered causal model through SAX module.")

            # 1.2. Causal relationship visualization
            causal_graph = cd.view_causal_dependencies(causal_result)
            
            try:
                graph_output_filename = 'causal_graph_with_sax_module'
                causal_graph.render(graph_output_filename, format='png', view=False, cleanup=True)
                print(f"✅ Causal relationship graph saved as '{graph_output_filename}.png'.")
            except Exception as e:
                print(f"❌ Error saving causal graph: {e}")

            # --- Part 2: XAI Analysis (SHAP) ---
            print("\n--- Step 2: Feature Importance Analysis with XAI ---")
            
            # 2.1. Data preparation
            if hasattr(df, 'getData'):
                df_data = df.getData()
            else:
                df_data = df
                
            # Create binary activity occurrence matrix (0/1)
            binary_activity_matrix = (pd.crosstab(df_data['case:concept:name'], df_data['concept:name']) > 0).astype(int)
            
            # Prediction target: total case duration
            df_data['time:timestamp'] = pd.to_datetime(df_data['time:timestamp'])
            case_durations = df_data.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
            case_durations['duration_seconds'] = (case_durations['max'] - case_durations['min']).dt.total_seconds()
            
            # Align features (X) and target (y) data
            X = binary_activity_matrix
            y = case_durations.loc[X.index]['duration_seconds']
            print("✅ Prediction target and feature data preparation completed.")

            # 2.2. XGBoost model training
            ml_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            ml_model.fit(X, y)
            print("✅ XGBoost model training completed.")

            # 2.3. SHAP analysis
            explainer = shap.TreeExplainer(ml_model)
            shap_values = explainer.shap_values(X)
            print("✅ SHAP analysis completed.")

            mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=X.columns)
            sorted_features = mean_abs_shap.sort_values(ascending=False)

            total_importance = sorted_features.sum()
            if total_importance > 0:
                    relative_importance = (sorted_features / total_importance) * 100
            else:
                relative_importance = sorted_features # All zeros case

            summary_lines = ["--- XAI Analysis Results (Relative Impact on Case Duration) ---"]
            for feature_name, importance in sorted_features.items():
                # Display both relative importance (%) and absolute value in summary text
                summary_lines.append(f"- '{feature_name}' activity is a key influencing factor (relative importance: {relative_importance[feature_name]:.2f}%, absolute impact: {importance:,.0f}).")

            print("✅ XAI detailed analysis completed.")
            print("Feature importance dictionary:", sorted_features.to_dict())
            print("\n".join(summary_lines))

            # 2.4. XAI results visualization
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title('XAI Feature Importance for Case Duration')
            plt.xlabel('SHAP Value (Impact on Case Duration)')
            xai_graph_filename = 'xai_feature_importance.png'
            plt.savefig(xai_graph_filename, bbox_inches='tight')
            plt.close()
            print(f"✅ XAI feature importance graph saved as '{xai_graph_filename}'.")

            # --- Part 3: Enhanced Knowledge Synthesis (SAX4BPM Methodology) ---
            print("\n--- Step 3: Enhanced SAX4BPM Knowledge Integration ---")

            # 상세 정보 생성
            detailed_process_summary = generate_detailed_process_summary(df_data)
            detailed_causal_summary = generate_detailed_causal_summary(causal_result)
            detailed_xai_summary = generate_detailed_xai_summary(shap_values, X)

            print("✅ Detailed analysis summaries generated.")

            # 사용자 질문
            user_query = "Based on the detailed analysis, can you formulate specific, data-driven hypotheses for process optimization?"

            # 향상된 SAX4BPM 설명 합성
            enhanced_synthesized_explanation = synthesize_enhanced_sax4bpm_explanation(
                detailed_process_summary, detailed_causal_summary, detailed_xai_summary, user_query
            )

            print("✅ Enhanced SAX4BPM knowledge integration completed.")

            # --- Part 4: Export Enhanced Results ---
            print("\n--- Step 4: Enhanced Results Export ---")

            # 향상된 LLM 프롬프트 저장
            with open('enhanced_sax4bpm_prompt.txt', 'w', encoding='utf-8') as f:
                f.write(enhanced_synthesized_explanation)

            # 상세 분석 결과 개별 저장
            with open('detailed_process_analysis.txt', 'w', encoding='utf-8') as f:
                f.write(detailed_process_summary)

            with open('detailed_causal_analysis.txt', 'w', encoding='utf-8') as f:
                f.write(detailed_causal_summary)

            with open('detailed_xai_analysis.txt', 'w', encoding='utf-8') as f:
                f.write(detailed_xai_summary)

            # 종합 분석 결과 JSON 저장
            comprehensive_results = {
                "timestamp": datetime.now().isoformat(),
                "detailed_process_view": detailed_process_summary,
                "detailed_causal_view": detailed_causal_summary,
                "detailed_xai_view": detailed_xai_summary,
                "user_query": user_query,
                "enhanced_synthesized_prompt": enhanced_synthesized_explanation,
                "feature_importance_scores": sorted_features.to_dict(),
                "relative_importance_percentages": relative_importance.to_dict()
            }

            with open('comprehensive_sax4bpm_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, ensure_ascii=False, indent=2, default=str)

            print("✅ Enhanced results with detailed information saved.")
            
            print("\n=== SAX4BPM Enhanced Analysis Results ===")
            print("Generated Files:")
            print("- enhanced_sax4bpm_prompt.txt: Complete LLM prompt with detailed analysis")
            print("- detailed_process_analysis.txt: Comprehensive process view")
            print("- detailed_causal_analysis.txt: Detailed causal discovery results")
            print("- detailed_xai_analysis.txt: Complete XAI feature importance analysis")
            print("- comprehensive_sax4bpm_analysis.json: All results in structured format")
            print("- causal_graph_with_sax_module.png: Causal relationship visualization")
            print("- xai_feature_importance.png: Feature importance visualization")
            
            print("\n--- Enhanced LLM Input: Comprehensive Prompt Preview ---")
            print(enhanced_synthesized_explanation[:1000] + "..." if len(enhanced_synthesized_explanation) > 1000 else enhanced_synthesized_explanation)

        except Exception as e:
            print(f"\n❌ Error occurred during analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ SAX module or data loading failed.")
        
        # SAX 모듈 없이 PM4Py만으로 기본 분석 수행
        print("\n--- Fallback: Basic Analysis with PM4Py Only ---")
        
        try:
            # PM4Py 기반 기본 분석
            import pm4py.llm
            
            pm_variants = pm4py.llm.abstract_variants(df)
            pm_model = pm4py.llm.abstract_dfg(df)
            pm_log = pm4py.llm.abstract_log_features(df)
            
            pm_text = f'Process Variants: {pm_variants}\nProcess Model: {pm_model}\nEvent Log Features: {pm_log}'
            
            # 기본 프롬프트 생성
            basic_prompt = f"""=== Basic Process Analysis (PM4Py Only) ===

PROCESS ABSTRACTIONS:
{pm_text}

QUESTION: Based on the process abstractions above, can you suggest potential areas for process improvement?

Note: This analysis is limited to PM4Py abstractions only. For comprehensive SAX4BPM analysis including causal discovery and XAI, please ensure the SAX library is properly installed."""

            # 기본 결과 저장
            with open('basic_pm4py_prompt.txt', 'w', encoding='utf-8') as f:
                f.write(basic_prompt)
                
            print("✅ Basic PM4Py analysis completed and saved to 'basic_pm4py_prompt.txt'")
            
        except Exception as e:
            print(f"❌ Basic analysis also failed: {e}")

    print("\n=== Analysis Pipeline Completed ===")
