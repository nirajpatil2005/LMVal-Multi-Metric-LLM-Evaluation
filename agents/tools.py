from langchain.tools import tool
from typing import Dict, Any
import time

@tool
def evaluate_response(question: str, ground_truth: str, response: str, metric: str, 
                     chain: Any, context: str = None) -> Dict[str, Any]:
    """Evaluate a response for a specific metric using LangChain"""
    start_time = time.time()
    
    try:
        # Prepare input based on metric type
        input_data = {
            "question": question,
            "ground_truth": ground_truth,
            "response": response
        }
        
        # Add context for context-based metrics (even if empty)
        if metric in ["context_precision", "context_recall"]:
            input_data["context"] = context if context else "No context provided."
        
        # Use invoke() instead of direct call to fix the tool calling issue
        result = chain.invoke(input_data)
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        return result
    except Exception as e:
        return {
            "score": 0,
            "explanation": f"Evaluation failed: {str(e)}",
            "processing_time": time.time() - start_time
        }