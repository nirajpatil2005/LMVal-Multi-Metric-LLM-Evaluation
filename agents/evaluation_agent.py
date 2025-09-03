from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from schemas.data_models import EvaluationRequest, EvaluationSummary, MetricType
from .graph_builder import EvaluationGraphBuilder

class EvaluationAgent:
    def __init__(self):
        self.graph_builder = None
    
    async def evaluate_async(self, request: EvaluationRequest) -> EvaluationSummary:
        """Evaluate questions asynchronously using LangGraph"""
        start_time = asyncio.get_event_loop().time()
        
        if len(request.questions) != len(request.ground_truths):
            raise ValueError("Questions and ground truths must have same length")
        
        if request.model_responses and len(request.questions) != len(request.model_responses):
            raise ValueError("Questions and model responses must have same length")
        
        # Initialize graph builder with API provider
        self.graph_builder = EvaluationGraphBuilder(
            model_name=request.judge_model,
            api_provider=request.api_provider.value
        )
        
        # Build evaluation graph
        graph = self.graph_builder.build_graph()
        
        # Process evaluations
        results = []
        with ThreadPoolExecutor(max_workers=request.max_concurrent) as executor:
            futures = []
            
            for i in range(len(request.questions)):
                state = {
                    "question": request.questions[i],
                    "ground_truth": request.ground_truths[i],
                    "model_response": request.model_responses[i] if request.model_responses else "",
                    "metrics": [m.value for m in request.metrics]
                }
                
                future = executor.submit(
                    self._run_evaluation,
                    graph,
                    state
                )
                futures.append(future)
            
            # Process with progress bar
            for future in tqdm(futures, desc="Evaluating responses"):
                try:
                    result = future.result()
                    results.append(result["final_result"])
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    # Add a failed result with default values
                    failed_result = {
                        "question": state["question"],
                        "ground_truth": state["ground_truth"],
                        "model_response": state["model_response"],
                        "metrics": {m: 0 for m in request.metrics},
                        "explanations": {m: f"Evaluation failed: {str(e)}" for m in request.metrics},
                        "processing_time": 0,
                        "overall_score": 0
                    }
                    results.append(failed_result)
        
        # Calculate summary
        avg_scores = self._calculate_average_scores(results, request.metrics)
        overall_score = self._calculate_overall_score(results)
        
        return EvaluationSummary(
            total_questions=len(request.questions),
            average_scores=avg_scores,
            individual_results=results,
            total_processing_time=asyncio.get_event_loop().time() - start_time,
            model_used=request.judge_model,
            api_provider=request.api_provider.value,
            overall_score=overall_score
        )
    
    def _run_evaluation(self, graph, state):
        """Run evaluation synchronously (for ThreadPoolExecutor)"""
        return graph.invoke(state)
    
    def _calculate_average_scores(self, results: List[Any], metrics: List[MetricType]) -> Dict[MetricType, float]:
        """Calculate average scores across all results"""
        avg_scores = {}
        for metric in metrics:
            scores = [result.metrics.get(metric.value, 0) for result in results]
            avg_scores[metric] = sum(scores) / len(scores) if scores else 0
        return avg_scores
    
    def _calculate_overall_score(self, results: List[Any]) -> float:
        """Calculate overall score across all results"""
        if not results:
            return 0
        
        overall_scores = [result.overall_score for result in results if hasattr(result, 'overall_score')]
        return sum(overall_scores) / len(overall_scores) if overall_scores else 0