import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import asyncio
import os
from typing import List, Dict, Any
from datetime import datetime, timezone

# Apply nest_asyncio to allow nested event loops
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Import your custom modules
try:
    from agents.evaluation_agent import EvaluationAgent
    from schemas.data_models import EvaluationRequest, MetricType, APIProvider
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
    metrics = list(scores.keys())
    values = list(scores.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        fillcolor='rgba(100, 149, 237, 0.3)',
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
    metrics = [m.capitalize() for m in scores.keys()]
    values = list(scores.values())

    # Create color scale based on score values - inverted for toxicity
    colors = []
    for metric, score in zip(metrics, values):
        if 'toxicity' in metric.lower():
            # For toxicity, lower is better (green), higher is worse (red)
            colors.append(f'hsl({int(120 * (100-score)/100)}, 70%, 50%)')
        else:
            # For other metrics, higher is better
            colors.append(f'hsl({int(120 * score/100)}, 70%, 50%)')

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
    if not results or not getattr(results[0], "metrics", None):
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

def get_score_color(metric: str, score: float) -> str:
    """Get color for a score based on metric type"""
    if 'toxicity' in metric.lower():
        # For toxicity, lower is better (green), higher is worse (red)
        return "green" if score <= 30 else "orange" if score <= 60 else "red"
    else:
        # For other metrics, higher is better
        return "green" if score >= 70 else "orange" if score >= 40 else "red"

def display_results(results):
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
        st.metric("API Provider", results.api_provider)
    with col5:
        st.metric("Overall Score", f"{results.overall_score:.1f}%")

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

        dist_fig = create_score_distribution_chart(results.individual_results)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
    else:
        st.warning("No metric scores available")

    # Detailed results
    st.subheader("📋 Detailed Results")
    if results.individual_results:
        tab1, tab2 = st.tabs(["Data Table", "Question Details"])

        with tab1:
            detailed_data = []
            for i, result in enumerate(results.individual_results):
                row = {
                    "ID": i + 1,
                    "Question": result.question[:50] + "..." if len(result.question) > 50 else result.question,
                    "Response": result.model_response[:50] + "..." if len(result.model_response) > 50 else result.model_response,
                    "Overall Score": f"{result.overall_score:.1f}%" if hasattr(result, 'overall_score') else "N/A",
                    "Time (s)": f"{result.processing_time:.2f}"
                }
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
                    "Overall Score": st.column_config.NumberColumn("Overall Score", width="medium"),
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
                        if hasattr(result, 'overall_score'):
                            st.metric("Overall Score", f"{result.overall_score:.1f}%")

                    with col2:
                        metrics_cols = st.columns(3)
                        metric_items = list(result.metrics.items())

                        for j, (metric, score) in enumerate(metric_items):
                            with metrics_cols[j % 3]:
                                # Use the correct color logic for each metric type
                                color = get_score_color(metric, score)
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

                        st.write("**Explanations:**")
                        if hasattr(result, 'explanations') and result.explanations:
                            selected_explanation = st.selectbox(
                                "Select metric explanation:",
                                options=list(result.explanations.keys()),
                                format_func=lambda x: x.capitalize(),
                                key=f"explanation_select_{i}"
                            )

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
            try:
                results_json = results.model_dump_json()
            except Exception:
                # Fallback serialization
                try:
                    results_json = json.dumps(results.__dict__, default=lambda o: getattr(o, "__dict__", str(o)), indent=2)
                except Exception:
                    results_json = "{}"

            st.download_button(
                "📊 Download JSON",
                data=results_json,
                file_name="evaluation_results.json",
                mime="application/json",
                use_container_width=True
            )

        with col2:
            csv_data = []
            for i, result in enumerate(results.individual_results):
                row = {
                    "ID": i + 1,
                    "Question": result.question,
                    "Ground Truth": result.ground_truth,
                    "Response": result.model_response,
                    "Overall Score": result.overall_score if hasattr(result, 'overall_score') else 0,
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
                <p>API Provider: {results.api_provider}</p>
                <p>Overall Score: {results.overall_score:.1f}%</p>

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

def build_request_object(questions: List[str], ground_truths: List[str], model_responses: List[str],
                         contexts: List[str], metrics: List[str], provider: str, judge_model: str,
                         max_concurrent: int):
    # Map provider to enum if available
    try:
        provider_enum = APIProvider.GROQ if provider.lower().startswith("groq") else APIProvider.OPENAI
    except Exception:
        provider_enum = provider

    # Try to instantiate EvaluationRequest robustly
    try:
        request = EvaluationRequest(
            questions=questions,
            ground_truths=ground_truths,
            model_responses=model_responses,
            metrics=[MetricType(m) for m in metrics],
            api_provider=provider_enum,
            judge_model=judge_model,
            max_concurrent=max_concurrent
        )
    except Exception:
        # Fallback to simple namespace-like object if model signature differs
        class SimpleRequest:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        request = SimpleRequest(
            questions=questions,
            ground_truths=ground_truths,
            model_responses=model_responses,
            metrics=metrics,
            api_provider=provider_enum,
            judge_model=judge_model,
            max_concurrent=max_concurrent
        )

    return request

def parse_json_file(uploaded_file):
    """Parse JSON file with different possible structures"""
    try:
        # Read and parse the file
        content = uploaded_file.getvalue()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        data = json.loads(content)
        
        questions_list = []
        truths_list = []
        responses_list = []
        contexts_list = []
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Check if it's the comprehensive_test_data.json format
            if "questions" in data and "ground_truths" in data:
                questions_list = data.get("questions", [])
                truths_list = data.get("ground_truths", [])
                responses_list = data.get("model_responses", [])
                contexts_list = data.get("contexts", [])
            else:
                # Try to extract from a single object
                item = {k.lower(): v for k, v in data.items()}
                q = item.get("question") or item.get("prompt") or item.get("input")
                gt = item.get("ground_truth") or item.get("groundtruth") or item.get("ground truth") or ""
                resp = item.get("model_response") or item.get("response") or item.get("answer") or ""
                ctx = item.get("context") or item.get("contexts") or ""
                
                if q:
                    questions_list.append(str(q))
                    truths_list.append(str(gt))
                    responses_list.append(str(resp))
                    contexts_list.append(str(ctx))
        
        elif isinstance(data, list):
            # Handle list of objects
            for item in data:
                if isinstance(item, dict):
                    item_lc = {k.lower(): v for k, v in item.items()}
                    q = item_lc.get("question") or item_lc.get("prompt") or item_lc.get("input")
                    gt = item_lc.get("ground_truth") or item_lc.get("groundtruth") or item_lc.get("ground truth") or ""
                    resp = item_lc.get("model_response") or item_lc.get("response") or item.lc.get("answer") or ""
                    ctx = item_lc.get("context") or item_lc.get("contexts") or ""
                    
                    if q:
                        questions_list.append(str(q))
                        truths_list.append(str(gt))
                        responses_list.append(str(resp))
                        contexts_list.append(str(ctx))
        
        return questions_list, truths_list, responses_list, contexts_list
        
    except Exception as e:
        st.error(f"Error parsing JSON file: {e}")
        return [], [], [], []

def main():
    st.title("🤖 LMVal: Multi-Metric LLM Evaluation")
    st.markdown("Advanced RAG pipeline evaluation using LangGraph and Groq/OpenAI")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_provider = st.radio(
            "API Provider",
            options=["groq", "openai"],
            index=0,
            horizontal=True
        )

        if api_provider == "groq":
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=os.getenv("GROQ_API_KEY", ""),
                help="Get from https://console.groq.com/"
            )
            if api_key:
                os.environ["GROQ_API_KEY"] = api_key
            
            judge_model = st.selectbox(
                "Judge Model",
                options=settings.AVAILABLE_GROQ_MODELS,
                index=0
            )
        else:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                help="Get from https://platform.openai.com/"
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            
            judge_model = st.selectbox(
                "Judge Model",
                options=settings.AVAILABLE_OPENAI_MODELS,
                index=0
            )

        selected_metrics = st.multiselect(
            "Evaluation Metrics",
            options=[m.value for m in MetricType],
            default=["accuracy", "faithfulness", "relevance"],
            help="Select metrics to evaluate. Some metrics may require additional context."
        )

        max_concurrent = st.slider(
            "Max Concurrent Evaluations",
            min_value=1,
            max_value=10,
            value=3,
            help="Higher values may cause rate limiting"
        )

        st.subheader("💾 Cache Settings")
        if st.button("Clear Cache", use_container_width=True):
            clear_cache()
            st.success("Cache cleared!")

        cache_stats = get_cache_stats()
        st.caption(f"Cache: {cache_stats['count']} items, {cache_stats['size'] / 1024 / 1024:.1f} MB")

        st.subheader("ℹ️ About")
        st.info("""
        This platform evaluates LLM responses using multiple metrics:
        - **Accuracy**: Comparison with ground truth (higher is better)
        - **Faithfulness**: Checks for hallucinations (higher is better)
        - **Relevance**: Response relevance to question (higher is better)
        - **Toxicity**: Detects harmful content (lower is better)
        - **Context Precision/Recall**: For RAG systems (higher is better)
        """)

    tab1, tab2, tab3 = st.tabs(["🏃‍♂️ Evaluate", "📊 Results", "📚 History"])

    # Evaluate tab
    with tab1:
        st.header("Run Evaluation")

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

            if any(metric in selected_metrics for metric in ["context_precision", "context_recall"]):
                contexts = st.text_area(
                    "Contexts (one per line, optional)",
                    height=100,
                    placeholder="France is a country...\nPlants use sunlight...",
                    help="Required for context precision/recall metrics"
                )
                contexts_list = [c.strip() for c in contexts.split('\n') if c.strip()]

            questions_list = [q.strip() for q in questions.split('\n') if q.strip()]
            truths_list = [g.strip() for g in ground_truths.split('\n') if g.strip()]
            responses_list = [r.strip() for r in model_responses.split('\n') if r.strip()]

        else:  # Upload JSON
            uploaded_file = st.file_uploader("Upload JSON file", type=["json"], 
                                           help="Upload a JSON file with questions, ground_truths, model_responses, and optionally contexts")
            
            if uploaded_file is not None:
                try:
                    questions_list, truths_list, responses_list, contexts_list = parse_json_file(uploaded_file)
                    
                    if questions_list:
                        st.success(f"Loaded {len(questions_list)} items from JSON")
                        
                        # Show preview
                        with st.expander("Preview loaded data"):
                            preview_data = {
                                "questions": questions_list[:3] + ["..."] if len(questions_list) > 3 else questions_list,
                                "ground_truths": truths_list[:3] + ["..."] if len(truths_list) > 3 else truths_list,
                                "model_responses": responses_list[:3] + ["..."] if responses_list and len(responses_list) > 3 else responses_list,
                                "contexts": contexts_list[:3] + ["..."] if contexts_list and len(contexts_list) > 3 else contexts_list
                            }
                            st.json(preview_data)
                    else:
                        st.warning("No valid data found in the JSON file")
                        
                except Exception as e:
                    st.error(f"Error processing JSON file: {e}")

        # Run evaluation button
        run_button = st.button("▶️ Run Evaluation", use_container_width=True, 
                              disabled=st.session_state.evaluation_in_progress)
        
        if run_button:
            if not questions_list:
                st.error("No questions provided.")
            elif len(questions_list) != len(truths_list):
                st.error("Number of questions and ground truths must match.")
            elif responses_list and len(questions_list) != len(responses_list):
                st.error("Number of questions and responses must match.")
            elif contexts_list and len(questions_list) != len(contexts_list):
                st.error("Number of questions and contexts must match for context-based metrics.")
            else:
                # Ensure we have responses (even if empty)
                if not responses_list:
                    responses_list = [""] * len(questions_list)
                
                # Ensure we have contexts (even if empty)
                if not contexts_list:
                    contexts_list = [""] * len(questions_list)
                
                # Build request object
                request = build_request_object(
                    questions=questions_list,
                    ground_truths=truths_list,
                    model_responses=responses_list,
                    contexts=contexts_list,
                    metrics=selected_metrics,
                    provider=api_provider,
                    judge_model=judge_model,
                    max_concurrent=max_concurrent
                )

                # Store evaluation parameters
                st.session_state.evaluation_params = {
                    "metrics": selected_metrics,
                    "provider": api_provider,
                    "judge_model": judge_model,
                    "max_concurrent": max_concurrent,
                    "num_items": len(questions_list),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                # Run evaluation
                st.session_state.evaluation_in_progress = True
                with st.spinner("Running evaluation..."):
                    results = run_evaluation_sync(request)
                    st.session_state.evaluation_in_progress = False

                if results:
                    st.success("Evaluation completed successfully!")
                    st.session_state.evaluation_results = results
                    
                    # Add to history
                    history_item = {
                        "id": len(st.session_state.evaluation_history) + 1,
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "params": st.session_state.evaluation_params,
                        "summary": {
                            "overall_score": getattr(results, "overall_score", None),
                            "total_questions": getattr(results, "total_questions", None)
                        },
                        "results": results
                    }
                    st.session_state.evaluation_history.insert(0, history_item)
                    st.session_state.show_results = True
                    st.session_state.active_tab = "Results"
                    st.rerun()
                else:
                    st.error("Evaluation failed. Please check your API keys and try again.")

        # Show current configuration
        if questions_list:
            st.info(f"Ready to evaluate {len(questions_list)} questions with {len(selected_metrics)} metrics using {judge_model}")

    # Results tab
    with tab2:
        st.header("Results")
        if st.session_state.show_results and st.session_state.evaluation_results:
            display_results(st.session_state.evaluation_results)
        else:
            st.info("No results to display. Run an evaluation from the Evaluate tab or load from History.")

    # History tab
    with tab3:
        st.header("Evaluation History")
        
        if not st.session_state.evaluation_history:
            st.info("No evaluation history yet. Run an evaluation first!")
        else:
            # Create a table for history
            history_data = []
            for item in st.session_state.evaluation_history:
                history_data.append({
                    "ID": item["id"],
                    "Timestamp": item["timestamp"],
                    "Questions": item["params"].get("num_items", "N/A"),
                    "Model": item["params"].get("judge_model", "N/A"),
                    "Provider": item["params"].get("provider", "N/A"),
                    "Overall Score": f"{item['summary'].get('overall_score', 0):.1f}%" if item['summary'].get('overall_score') is not None else "N/A"
                })
            
            # Display history as a table
            history_df = pd.DataFrame(history_data)
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID": st.column_config.NumberColumn("Run #", width="small"),
                    "Timestamp": st.column_config.DatetimeColumn("Time", width="medium"),
                    "Questions": st.column_config.NumberColumn("Questions", width="small"),
                    "Model": st.column_config.TextColumn("Model", width="medium"),
                    "Provider": st.column_config.TextColumn("Provider", width="small"),
                    "Overall Score": st.column_config.TextColumn("Score", width="small")
                }
            )
            
            # Action buttons for each history item
            selected_run = st.selectbox(
                "Select a run to view or manage:",
                options=[f"Run #{item['id']} - {item['timestamp']}" for item in st.session_state.evaluation_history],
                index=0
            )
            
            # Extract run ID from selection
            run_id = int(selected_run.split("#")[1].split(" ")[0]) if selected_run else None
            
            if run_id:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 View Results", use_container_width=True):
                        # Find the selected run
                        selected_item = next((item for item in st.session_state.evaluation_history if item["id"] == run_id), None)
                        if selected_item:
                            st.session_state.evaluation_results = selected_item["results"]
                            st.session_state.show_results = True
                            st.session_state.active_tab = "Results"
                            st.rerun()
                
                with col2:
                    if st.button("📥 Export Results", use_container_width=True):
                        selected_item = next((item for item in st.session_state.evaluation_history if item["id"] == run_id), None)
                        if selected_item and hasattr(selected_item["results"], 'model_dump_json'):
                            results_json = selected_item["results"].model_dump_json()
                            st.download_button(
                                "Download JSON",
                                data=results_json,
                                file_name=f"evaluation_run_{run_id}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                
                with col3:
                    if st.button("🗑️ Delete Run", use_container_width=True):
                        st.session_state.evaluation_history = [
                            item for item in st.session_state.evaluation_history if item["id"] != run_id
                        ]
                        st.success(f"Deleted run #{run_id}")
                        st.rerun()
            
            # Clear all history button
            if st.button("Clear All History", use_container_width=True, type="secondary"):
                st.session_state.evaluation_history = []
                st.success("All history cleared")
                st.rerun()

if __name__ == "__main__":
    main()