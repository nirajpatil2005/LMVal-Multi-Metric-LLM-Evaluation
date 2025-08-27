from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
from typing_extensions import TypedDict
import asyncio

from schemas.data_models import EvaluationResult
from chains.evaluation_chains import EvaluationChains
from .tools import evaluate_response

class EvaluationState(TypedDict):
    question: str
    ground_truth: str
    model_response: str
    context: str
    metrics: List[str]
    results: Dict[str, Any]
    current_metric: str
    final_result: EvaluationResult

class EvaluationGraphBuilder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.evaluation_chains = EvaluationChains(model_name)
    
    def build_graph(self):
        """Build LangGraph workflow for evaluation"""
        workflow = StateGraph(EvaluationState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("evaluate_metric", self._evaluate_metric_node)
        workflow.add_node("aggregate_results", self._aggregate_results_node)
        
        # Define edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "evaluate_metric")
        workflow.add_conditional_edges(
            "evaluate_metric",
            self._should_continue,
            {
                "continue": "evaluate_metric",
                "done": "aggregate_results"
            }
        )
        workflow.add_edge("aggregate_results", END)
        
        return workflow.compile()
    
    def _initialize_node(self, state: EvaluationState) -> EvaluationState:
        """Initialize evaluation state"""
        return {
            **state,
            "results": {},
            "current_metric": state["metrics"][0] if state["metrics"] else ""
        }
    
    def _evaluate_metric_node(self, state: EvaluationState) -> EvaluationState:
        """Evaluate a single metric"""
        metric = state["current_metric"]
        chain = self.evaluation_chains.create_evaluation_chain(metric)
        
        # Prepare tool input
        tool_input = {
            "question": state["question"],
            "ground_truth": state["ground_truth"],
            "response": state["model_response"],
            "metric": metric,
            "chain": chain
        }
        
        # Add context for context-based metrics
        if metric in ["context_precision", "context_recall"] and "context" in state:
            tool_input["context"] = state["context"]
        
        # Fix: Use the tool correctly with proper arguments
        result = evaluate_response.invoke(tool_input)
        
        # Update results
        results = state.get("results", {})
        results[metric] = result
        
        # Move to next metric
        current_index = state["metrics"].index(metric)
        next_index = current_index + 1
        
        return {
            **state,
            "results": results,
            "current_metric": state["metrics"][next_index] if next_index < len(state["metrics"]) else None
        }
    
    def _should_continue(self, state: EvaluationState) -> str:
        """Determine if we should continue evaluating metrics"""
        if state["current_metric"] is None:
            return "done"
        return "continue"
    
    def _aggregate_results_node(self, state: EvaluationState) -> EvaluationState:
        """Aggregate results into final format"""
        metrics_scores = {}
        explanations = {}
        total_time = 0
        
        for metric, result in state["results"].items():
            metrics_scores[metric] = result.get("score", 0)
            explanations[metric] = result.get("explanation", "")
            total_time += result.get("processing_time", 0)
        
        final_result = EvaluationResult(
            question=state["question"],
            ground_truth=state["ground_truth"],
            model_response=state["model_response"],
            metrics=metrics_scores,
            explanations=explanations,
            processing_time=total_time
        )
        
        return {**state, "final_result": final_result}