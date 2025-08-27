import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import asyncio
import os
import time
import numpy as np
from typing import List, Dict, Any

# Apply nest_asyncio to allow nested event loops
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Import your custom modules
try:
    from agents.evaluation_agent import EvaluationAgent
    from schemas.data_models import EvaluationRequest, MetricType
    from config import settings
    from utils.cache_manager import clear_cache, get_cache_stats
except ImportError as e:
    st.error(f"Import error: {e}. Please make sure all required modules are available.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="LLM Evaluation Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []
if "evaluation_in_progress" not in st.session_state:
    st.session_state.evaluation_in_progress = False
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Evaluate"
if "evaluation_params" not in st.session_state:
    st.session_state.evaluation_params = {}
if "show_results" not in st.session_state:
    st.session_state.show_results = False

def run_evaluation_sync(request: EvaluationRequest):
    """Run evaluation synchronously with proper event loop handling"""
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        agent = EvaluationAgent()
        result = loop.run_until_complete(agent.evaluate_async(request))
        loop.close()
        return result
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return None

def create_metric_radar_chart(scores: Dict[str, float]) -> go.Figure:
    """Create an interactive radar chart for metrics"""
    metrics = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the shape
        theta=metrics + [metrics[0]],  # Close the shape
        fill='toself',
        fillcolor='rgba(100, 149, 237, 0.3)',  # Cornflower blue with opacity
        line=dict(color='rgba(100, 149, 237, 0.8)', width=3),
        name='Metrics Score',
        hoverinfo='text',
        hovertext=[f'{metric}: {score:.1f}%' for metric, score in zip(metrics, values)]
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                tickangle=0,
                tickvals=[0, 20, 40, 60, 80, 100],
                ticktext=['0%', '20%', '40%', '60%', '80%', '100%']
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                rotation=90
            )
        ),
        showlegend=False,
        title=dict(
            text="Performance Metrics Radar",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_metric_bar_chart(scores: Dict[str, float]) -> go.Figure:
    """Create an interactive bar chart for metrics"""
    metrics = [m.capitalize() for m in scores.keys()]
    values = list(scores.values())
    
    # Create color scale based on score values
    colors = [f'hsl({int(120 * score/100)}, 70%, 50%)' for score in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color=colors,
        marker_line=dict(color='rgba(0,0,0,0.3)', width=1),
        text=[f'{v:.1f}%' for v in values],
        textposition='auto',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Average Scores by Metric",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Evaluation Metric",
            tickangle=45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="Score (%)",
            range=[0, 100],
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=['0%', '20%', '40%', '60%', '80%', '100%']
        ),
        height=450,
        margin=dict(l=50, r=50, t=80, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_score_distribution_chart(results: List[Any]) -> go.Figure:
    """Create distribution charts for each metric"""
    if not results or not results[0].metrics:
        return None
    
    metrics = list(results[0].metrics.keys())
    fig = go.Figure()
    
    for metric in metrics:
        scores = [getattr(r, 'metrics', {}).get(metric, 0) for r in results]
        
        fig.add_trace(go.Violin(
            y=scores,
            name=metric.capitalize(),
            box_visible=True,
            meanline_visible=True,
            points="all",
            hoverinfo='y',
            opacity=0.7
        ))
    
    fig.update_layout(
        title=dict(
            text="Score Distribution by Metric",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        yaxis=dict(
            title="Score (%)",
            range=[0, 100],
            tickvals=[0, 20, 40, 60, 80, 100]
        ),
        xaxis=dict(title="Metric"),
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def display_results(results):
    """Display evaluation results in the UI"""
    if not results:
        st.error("No results to display")
        return
    
    if not hasattr(results, 'individual_results') or not results.individual_results:
        st.warning("No individual results available")
        return
    
    # Summary
    st.subheader("📊 Evaluation Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Questions", results.total_questions)
    with col2:
        st.metric("Total Time", f"{results.total_processing_time:.1f}s")
    with col3:
        st.metric("Model Used", results.model_used)
    with col4:
        avg_score = sum(results.average_scores.values()) / len(results.average_scores) if results.average_scores else 0
        st.metric("Overall Score", f"{avg_score:.1f}%")
    with col5:
        st.metric("Metrics Evaluated", len(results.average_scores))
    
    # Metrics visualization
    st.subheader("📈 Performance Metrics")
    
    if results.average_scores:
        col1, col2 = st.columns(2)
        
        with col1:
            bar_fig = create_metric_bar_chart(results.average_scores)
            st.plotly_chart(bar_fig, use_container_width=True)
        
        with col2:
            radar_fig = create_metric_radar_chart(results.average_scores)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Score distribution
        dist_fig = create_score_distribution_chart(results.individual_results)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
    else:
        st.warning("No metric scores available")
    
    # Detailed results
    st.subheader("📋 Detailed Results")
    if results.individual_results:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Data Table", "Question Details"])
        
        with tab1:
            detailed_data = []
            for i, result in enumerate(results.individual_results):
                row = {
                    "ID": i + 1,
                    "Question": result.question[:50] + "..." if len(result.question) > 50 else result.question,
                    "Response": result.model_response[:50] + "..." if len(result.model_response) > 50 else result.model_response,
                    "Time (s)": f"{result.processing_time:.2f}"
                }
                # Add metric scores
                for metric, score in result.metrics.items():
                    row[metric.capitalize()] = f"{score:.1f}%"
                detailed_data.append(row)
            
            st.dataframe(
                detailed_data, 
                use_container_width=True,
                height=400,
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Question": st.column_config.TextColumn("Question", width="large"),
                    "Response": st.column_config.TextColumn("Response", width="large"),
                }
            )
        
        with tab2:
            for i, result in enumerate(results.individual_results):
                with st.expander(f"Question {i+1}: {result.question[:70]}{'...' if len(result.question) > 70 else ''}", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Question:**")
                        st.info(result.question)
                        
                        st.write("**Ground Truth:**")
                        st.success(result.ground_truth)
                        
                        st.write("**Model Response:**")
                        st.info(result.model_response)
                        
                        st.metric("Processing Time", f"{result.processing_time:.2f}s")
                    
                    with col2:
                        # Metrics scores
                        metrics_cols = st.columns(3)
                        metric_items = list(result.metrics.items())
                        
                        for j, (metric, score) in enumerate(metric_items):
                            with metrics_cols[j % 3]:
                                color = "green" if score >= 70 else "orange" if score >= 40 else "red"
                                st.markdown(f"""
                                <div style="background-color: rgba(240, 242, 246, 0.5); 
                                            padding: 15px; 
                                            border-radius: 10px; 
                                            border-left: 4px solid {color};
                                            margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: {color};">{metric.capitalize()}</h4>
                                    <h2 style="margin: 5px 0; color: {color};">{score:.1f}%</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Explanations - Use a selectbox instead of nested expanders
                        st.write("**Explanations:**")
                        if hasattr(result, 'explanations') and result.explanations:
                            # Create a dropdown to select which explanation to view
                            selected_explanation = st.selectbox(
                                "Select metric explanation:",
                                options=list(result.explanations.keys()),
                                format_func=lambda x: x.capitalize(),
                                key=f"explanation_select_{i}"
                            )
                            
                            # Display the selected explanation
                            st.text_area(
                                f"{selected_explanation.capitalize()} Explanation",
                                value=result.explanations[selected_explanation],
                                height=150,
                                key=f"explanation_text_{i}_{selected_explanation}",
                                disabled=True
                            )
                        else:
                            st.info("No explanations available for this question")
        
        # Export buttons
        st.subheader("💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            results_json = results.model_dump_json()
            st.download_button(
                "📊 Download JSON",
                data=results_json,
                file_name="evaluation_results.json",
                mime="application/json",
                use_container_width=True
            )
            
        with col2:
            # Create CSV data
            csv_data = []
            for i, result in enumerate(results.individual_results):
                row = {
                    "ID": i + 1,
                    "Question": result.question,
                    "Ground Truth": result.ground_truth,
                    "Response": result.model_response,
                    "Time (s)": result.processing_time
                }
                for metric, score in result.metrics.items():
                    row[metric.capitalize()] = score
                if hasattr(result, 'explanations'):
                    for metric, explanation in result.explanations.items():
                        row[f"{metric.capitalize()} Explanation"] = explanation
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            csv = df.to_csv(index=False)
            st.download_button(
                "📋 Download CSV",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col3:
            # Create HTML report
            html_content = f"""
            <html>
            <head>
                <title>LLM Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px; border-radius: 5px; }}
                    .score {{ font-size: 24px; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>LLM Evaluation Report</h1>
                    <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <h2>Summary</h2>
                <p>Total Questions: {results.total_questions}</p>
                <p>Total Time: {results.total_processing_time:.1f}s</p>
                <p>Model Used: {results.model_used}</p>
                
                <h2>Average Scores</h2>
                {"".join([f'<div class="metric"><h3>{m.capitalize()}</h3><div class="score">{s:.1f}%</div></div>' for m, s in results.average_scores.items()])}
            </body>
            </html>
            """
            
            st.download_button(
                "🌐 Download HTML Report",
                data=html_content,
                file_name="evaluation_report.html",
                mime="text/html",
                use_container_width=True
            )
    else:
        st.warning("No individual results available")

def main():
    st.title("🤖 LMVal: Multi-Metric LLM Evaluation")
    st.markdown("Advanced RAG pipeline evaluation using LangGraph and Groq")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get from https://console.groq.com/"
        )
        
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
        # Model selection
        judge_model = st.selectbox(
            "Judge Model",
            options=settings.AVAILABLE_MODELS,
            index=0
        )
        
        # Metrics selection
        selected_metrics = st.multiselect(
            "Evaluation Metrics",
            options=[m.value for m in MetricType],
            default=["accuracy", "faithfulness", "relevance"],
            help="Select metrics to evaluate. Some metrics may require additional context."
        )
        
        # Concurrency
        max_concurrent = st.slider(
            "Max Concurrent Evaluations",
            min_value=1,
            max_value=10,
            value=3,
            help="Higher values may cause rate limiting"
        )
        
        # Cache settings
        st.subheader("💾 Cache Settings")
        if st.button("Clear Cache", use_container_width=True):
            clear_cache()
            st.success("Cache cleared!")
        
        cache_stats = get_cache_stats()
        st.caption(f"Cache: {cache_stats['count']} items, {cache_stats['size'] / 1024 / 1024:.1f} MB")
        
        # Info section
        st.subheader("ℹ️ About")
        st.info("""
        This platform evaluates LLM responses using multiple metrics:
        - **Accuracy**: Comparison with ground truth
        - **Faithfulness**: Checks for hallucinations
        - **Relevance**: Response relevance to question
        - **Toxicity**: Detects harmful content
        - **Context Precision/Recall**: For RAG systems
        """)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🏃‍♂️ Evaluate", "📊 Results", "📚 History"])
    
    with tab1:
        st.header("Run Evaluation")
        
        # Input method
        input_method = st.radio(
            "Input Method",
            ["Manual Input", "Upload JSON"],
            horizontal=True
        )
        
        questions_list = []
        truths_list = []
        responses_list = []
        contexts_list = []
        
        if input_method == "Manual Input":
            col1, col2 = st.columns(2)
            
            with col1:
                questions = st.text_area(
                    "Questions (one per line)",
                    height=150,
                    placeholder="What is the capital of France?\nHow does photosynthesis work?",
                    help="Enter each question on a new line"
                )
            
            with col2:
                ground_truths = st.text_area(
                    "Ground Truths (one per line)",
                    height=150,
                    placeholder="Paris\nPhotosynthesis converts sunlight to energy.",
                    help="Enter ground truth for each question"
                )
            
            model_responses = st.text_area(
                "Model Responses (one per line)",
                height=150,
                placeholder="Paris is the capital.\nPhotosynthesis uses sunlight.",
                help="Enter model response for each question"
            )
            
            # Context input for context-based metrics
            if any(metric in selected_metrics for metric in ["context_precision", "context_recall"]):
                contexts = st.text_area(
                    "Contexts (one per line, optional)",
                    height=100,
                    placeholder="France is a country...\nPlants use sunlight...",
                    help="Required for context precision/recall metrics"
                )
                contexts_list = [c.strip() for c in contexts.split('\n') if c.strip()] if contexts else []
            
            questions_list = [q.strip() for q in questions.split('\n') if q.strip()]
            truths_list = [t.strip() for t in ground_truths.split('\n') if t.strip()]
            responses_list = [r.strip() for r in model_responses.split('\n') if r.strip()] if model_responses else []
            
        else:  # Upload JSON
            uploaded_file = st.file_uploader("Upload JSON", type="json")
            if uploaded_file:
                try:
                    data = json.load(uploaded_file)
                    questions_list = data.get("questions", [])
                    truths_list = data.get("ground_truths", [])
                    responses_list = data.get("model_responses", [])
                    contexts_list = data.get("contexts", [])
                    
                    # Display preview
                    st.subheader("Data Preview")
                    preview_data = {
                        "questions": questions_list[:3] + ["..."] if len(questions_list) > 3 else questions_list,
                        "ground_truths": truths_list[:3] + ["..."] if len(truths_list) > 3 else truths_list,
                        "model_responses": responses_list[:3] + ["..."] if responses_list and len(responses_list) > 3 else responses_list
                    }
                    st.json(preview_data)
                    
                except Exception as e:
                    st.error(f"Error loading JSON file: {e}")
        
        # Validation for context-based metrics
        if any(metric in selected_metrics for metric in ["context_precision", "context_recall"]):
            if not contexts_list:
                st.warning("⚠️ Context precision/recall metrics require context input. Please provide contexts.")
            elif len(contexts_list) != len(questions_list):
                st.error("❌ Number of contexts must match number of questions for context-based metrics")
        
        # Run evaluation button
        if st.button("🚀 Run Evaluation", use_container_width=True, disabled=st.session_state.evaluation_in_progress):
            if not api_key:
                st.error("❌ Please enter Groq API key")
                return
            
            if not questions_list or not truths_list:
                st.error("❌ Please provide questions and ground truths")
                return
            
            if len(questions_list) != len(truths_list):
                st.error("❌ Questions and ground truths must match in count")
                return
            
            if responses_list and len(questions_list) != len(responses_list):
                st.error("❌ Questions and responses must match in count")
                return
            
            # Store evaluation parameters in session state
            st.session_state.evaluation_params = {
                "questions": questions_list,
                "ground_truths": truths_list,
                "responses": responses_list if responses_list else [""] * len(questions_list),
                "contexts": contexts_list if contexts_list else [""] * len(questions_list),
                "metrics": [MetricType(m) for m in selected_metrics],
                "model": judge_model,
                "concurrent": max_concurrent
            }
            
            st.session_state.evaluation_in_progress = True
            st.session_state.show_results = False
            st.rerun()
    
    # Run evaluation if parameters are set and evaluation is in progress
    if (st.session_state.evaluation_in_progress and 
        "evaluation_params" in st.session_state):
        
        params = st.session_state.evaluation_params
        
        with st.spinner("🚀 Running evaluation with LangGraph..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create evaluation request
                request = EvaluationRequest(
                    questions=params["questions"],
                    ground_truths=params["ground_truths"],
                    model_responses=params["responses"],
                    metrics=params["metrics"],
                    judge_model=params["model"],
                    max_concurrent=params["concurrent"]
                )
                
                # Update progress
                progress_bar.progress(30)
                status_text.text("Initializing evaluation agent...")
                
                # Run evaluation synchronously (avoids thread issues)
                results = run_evaluation_sync(request)
                
                if results:
                    progress_bar.progress(100)
                    status_text.text("✅ Evaluation completed!")
                    
                    # Store results in session state
                    st.session_state.evaluation_results = results
                    st.session_state.evaluation_history.append({
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "results": results,
                        "config": {
                            "model": params["model"],
                            "metrics": [m.value for m in params["metrics"]],
                            "question_count": len(params["questions"])
                        }
                    })
                    
                    # Set flag to show results
                    st.session_state.show_results = True
                    st.session_state.evaluation_in_progress = False
                    st.session_state.active_tab = "Results"
                    
                    # Rerun to show results
                    st.rerun()
                else:
                    st.error("❌ Evaluation failed to return results")
                    st.session_state.evaluation_in_progress = False
                    
            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")
                st.session_state.evaluation_in_progress = False
    
    with tab2:
        st.header("Results")
        
        # Check if results exist before trying to access them
        if st.session_state.evaluation_results:
            display_results(st.session_state.evaluation_results)
        else:
            st.info("📝 No evaluation results yet. Run an evaluation first!")
            
            if st.session_state.evaluation_in_progress:
                with st.status("⏳ Evaluation in progress...", expanded=True) as status:
                    st.write("Processing questions...")
                    st.write("This may take a few minutes depending on the number of questions")
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.1)
                        progress.progress(i + 1)
    
    with tab3:
        st.header("Evaluation History")
        
        if st.session_state.evaluation_history:
            for i, history in enumerate(reversed(st.session_state.evaluation_history)):
                eval_num = len(st.session_state.evaluation_history) - i
                with st.expander(f"Evaluation {eval_num} - {history['timestamp'][:19]}", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Model", history["config"]["model"])
                    col2.metric("Questions", history["config"]["question_count"])
                    col3.metric("Metrics", len(history["config"]["metrics"]))
                    
                    # Calculate average score
                    avg_score = sum(history["results"].average_scores.values()) / len(history["results"].average_scores) if history["results"].average_scores else 0
                    col4.metric("Avg Score", f"{avg_score:.1f}%")
                    
                    if st.button(f"📊 Load Results", key=f"load_{i}", use_container_width=True):
                        st.session_state.evaluation_results = history["results"]
                        st.session_state.active_tab = "Results"
                        st.rerun()
                        
                    if st.button(f"🗑️ Delete", key=f"delete_{i}", use_container_width=True):
                        st.session_state.evaluation_history.pop(len(st.session_state.evaluation_history) - i - 1)
                        st.rerun()
        else:
            st.info("📚 No evaluation history yet. Run an evaluation first!")

if __name__ == "__main__":
    main()