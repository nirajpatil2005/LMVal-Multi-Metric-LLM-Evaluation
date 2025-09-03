from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from enum import Enum

class MetricType(str, Enum):
    ACCURACY = "accuracy"
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    TOXICITY = "toxicity"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"

class APIProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"

class EvaluationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    questions: List[str] = Field(..., description="Questions to evaluate")
    ground_truths: List[str] = Field(..., description="Ground truth answers")
    model_responses: Optional[List[str]] = Field(None, description="Model responses")
    metrics: List[MetricType] = Field(default=["accuracy", "faithfulness", "relevance"])
    judge_model: str = Field(default="openai/gpt-oss-20b")
    max_concurrent: int = Field(default=5, description="Max concurrent evaluations")
    api_provider: APIProvider = Field(default=APIProvider.GROQ, description="API provider for evaluation")

class EvaluationResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    question: str
    ground_truth: str
    model_response: str
    metrics: Dict[MetricType, float]
    explanations: Dict[MetricType, str]
    processing_time: float
    overall_score: float = Field(..., description="Overall weighted score (0-100)")

class EvaluationSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    total_questions: int
    average_scores: Dict[MetricType, float]
    individual_results: List[EvaluationResult]
    total_processing_time: float
    model_used: str
    api_provider: str
    overall_score: float = Field(..., description="Overall weighted score across all questions")