from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import json
from typing import Dict, Any

from .prompt_templates import PROMPT_MAP, EVALUATION_SYSTEM_MESSAGE
from config import settings

class EvaluationChains:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.DEFAULT_MODEL
        self.llm = ChatGroq(
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=500
        )
    
    def create_evaluation_chain(self, metric: str):
        """Create a LangChain chain for a specific evaluation metric"""
        prompt = PROMPT_MAP.get(metric)
        if not prompt:
            raise ValueError(f"Unknown metric: {metric}")
        
        chain = (
            RunnablePassthrough()
            | prompt
            | self.llm
            | StrOutputParser()
            | self._parse_json_response
        )
        return chain
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            return {"score": 0, "explanation": "Invalid response format"}
        except json.JSONDecodeError:
            return {"score": 0, "explanation": "Failed to parse JSON response"}