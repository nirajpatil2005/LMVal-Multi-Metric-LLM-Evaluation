from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from schemas.data_models import EvaluationRequest, EvaluationSummary
from .graph_builder import EvaluationGraphBuilder
from config import settings

class EvaluationAgent:
    def __init__(self):
        self.graph_builder = EvaluationGraphBuilder()
    
    async def evaluate_async(self, request: EvaluationRequest) -> EvaluationSummary:
        """Evaluate questions asynchronously using LangGraph"""
        start_time = asyncio.get_event_loop().time()
        
        if len(request.questions) != len(request.ground_truths):
            raise ValueError("Questions and ground truths must have same length")
        
        if request.model_responses and len(request.questions) != len(request.model_responses):
            raise ValueError("Questions and model responses must have same length")
        
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
        
        # Calculate summary
        avg_scores = self._calculate_average_scores(results, request.metrics)
        
        return EvaluationSummary(
            total_questions=len(request.questions),
            average_scores=avg_scores,
            individual_results=results,
            total_processing_time=asyncio.get_event_loop().time() - start_time,
            model_used=request.judge_model
        )
    
    def _run_evaluation(self, graph, state):
        """Run evaluation synchronously (for ThreadPoolExecutor)"""
        return graph.invoke(state)
    
    def _calculate_average_scores(self, results: List[Any], metrics: List[str]) -> Dict[str, float]:
        """Calculate average scores across all results"""
        avg_scores = {}
        for metric in metrics:
            scores = [result.metrics.get(metric, 0) for result in results]
            avg_scores[metric] = sum(scores) / len(scores) if scores else 0
        return avg_scores