import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from .config import PipelineConfig
from .query_generator import Query
from .trajectory_collector import Trajectory
from .trajectory_evaluator import TrajectoryEvaluation, CriticLabel

logger = logging.getLogger(__name__)

@dataclass
class QueryGuidance:
    query_id: str
    steps: List[Dict[str, Any]]
    key_actions: List[str]
    critical_points: List[str]
    common_errors: List[str]
    confidence: float
    metadata: Dict[str, Any] = None

class GuidanceGenerator:
    def __init__(self, config: PipelineConfig, model_client=None):
        self.config = config
        self.model_client = model_client
        self.guidance_cache: Dict[str, QueryGuidance] = {}
        
    async def generate_guidance(self, 
                               query: Query, 
                               failed_trajectory: Optional[Trajectory] = None,
                               evaluation: Optional[TrajectoryEvaluation] = None) -> QueryGuidance:
        
        if query.id in self.guidance_cache and not failed_trajectory:
            logger.info(f"Using cached guidance for query {query.id}")
            return self.guidance_cache[query.id]
            
        action_descriptions = await self._generate_action_descriptions(query, failed_trajectory)
        
        quality_validated = await self._validate_quality(action_descriptions, failed_trajectory)
        
        key_steps = self._extract_key_steps(query, failed_trajectory, evaluation)
        
        guidance = self._synthesize_guidance(query, action_descriptions, key_steps, evaluation)
        
        self.guidance_cache[query.id] = guidance
        
        logger.info(f"Generated guidance for query {query.id} with {len(guidance.steps)} steps")
        
        return guidance
        
    async def _generate_action_descriptions(self, query: Query, failed_trajectory: Optional[Trajectory]) -> List[Dict]:
        descriptions = []
        
        if self.model_client and failed_trajectory:
            for step in failed_trajectory.steps:
                description = await self._describe_action_effect(step)
                descriptions.append({
                    "step_index": step.index,
                    "action": step.action,
                    "description": description,
                    "success": step.success
                })
        else:
            descriptions = self._generate_default_descriptions(query)
            
        return descriptions
        
    async def _describe_action_effect(self, step) -> str:
        if self.model_client:
            prompt = f"""
            Describe the effect of this action:
            Action: {step.action}
            Success: {step.success}
            Error: {step.error}
            
            Provide a clear description of what this action does and its expected result.
            """
            
            return await self._call_model_for_description(prompt)
        else:
            action_type = step.action.get("type", "unknown")
            if step.success:
                return f"Successfully executed {action_type} action"
            else:
                return f"Failed to execute {action_type} action: {step.error}"
                
    async def _call_model_for_description(self, prompt: str) -> str:
        return "This action clicks on the navigation menu to reveal additional options"
        
    def _generate_default_descriptions(self, query: Query) -> List[Dict]:
        descriptions = []
        
        for i, page in enumerate(query.path):
            descriptions.append({
                "step_index": i,
                "action": {"type": "navigate", "target": page},
                "description": f"Navigate to {page}",
                "success": True
            })
            
        return descriptions
        
    async def _validate_quality(self, descriptions: List[Dict], failed_trajectory: Optional[Trajectory]) -> bool:
        if not self.model_client or not failed_trajectory:
            return True
            
        for i, desc in enumerate(descriptions):
            if i < len(failed_trajectory.steps):
                step = failed_trajectory.steps[i]
                
                consistency = await self._check_consistency(desc["description"], step)
                
                if not consistency:
                    logger.warning(f"Inconsistent description for step {i}")
                    desc["description"] = self._fallback_description(step)
                    
        return True
        
    async def _check_consistency(self, description: str, step) -> bool:
        if self.model_client:
            prompt = f"""
            Check if this description matches the actual action:
            Description: {description}
            Actual action: {step.action}
            Actual result: {"Success" if step.success else f"Failed: {step.error}"}
            
            Is the description consistent with the actual action? (YES/NO)
            """
            
            response = await self._call_model_for_consistency(prompt)
            return response.upper() == "YES"
        else:
            return True
            
    async def _call_model_for_consistency(self, prompt: str) -> str:
        return "YES"
        
    def _fallback_description(self, step) -> str:
        return f"Execute {step.action.get('type', 'unknown')} action at step {step.index}"
        
    def _extract_key_steps(self, query: Query, failed_trajectory: Optional[Trajectory], evaluation: Optional[TrajectoryEvaluation]) -> List[Dict]:
        key_steps = []
        
        if evaluation and evaluation.step_evaluations:
            for i, step_eval in enumerate(evaluation.step_evaluations):
                if step_eval.annotation == CriticLabel.HARMFUL:
                    key_steps.append({
                        "index": i,
                        "type": "critical_error",
                        "description": step_eval.summary,
                        "action_needed": "Avoid this action or retry with different parameters"
                    })
                elif step_eval.annotation == CriticLabel.GOOD and step_eval.confidence > 0.8:
                    key_steps.append({
                        "index": i,
                        "type": "key_success",
                        "description": step_eval.summary,
                        "action_needed": "Ensure this step is executed correctly"
                    })
                    
        if not key_steps and query.slots:
            for slot_name, slot_value in query.slots.items():
                key_steps.append({
                    "index": -1,
                    "type": "required_input",
                    "description": f"Provide {slot_name}",
                    "action_needed": f"Enter value: {slot_value}"
                })
                
        return key_steps
        
    def _synthesize_guidance(self, query: Query, descriptions: List[Dict], key_steps: List[Dict], evaluation: Optional[TrajectoryEvaluation]) -> QueryGuidance:
        
        steps = []
        for desc in descriptions:
            step_guidance = {
                "index": desc["step_index"],
                "action": desc["action"],
                "description": desc["description"],
                "expected_result": "Success" if desc.get("success", True) else "May fail - have alternative ready"
            }
            
            for key_step in key_steps:
                if key_step["index"] == desc["step_index"]:
                    step_guidance["critical"] = True
                    step_guidance["special_instruction"] = key_step["action_needed"]
                    
            steps.append(step_guidance)
            
        key_actions = [step["description"] for step in key_steps if step["type"] == "key_success"]
        
        critical_points = [step["description"] for step in key_steps if step["type"] == "critical_error"]
        
        common_errors = self._identify_common_errors(failed_trajectory, evaluation)
        
        confidence = self._calculate_guidance_confidence(descriptions, key_steps, evaluation)
        
        guidance = QueryGuidance(
            query_id=query.id,
            steps=steps,
            key_actions=key_actions,
            critical_points=critical_points,
            common_errors=common_errors,
            confidence=confidence,
            metadata={
                "query_difficulty": query.difficulty,
                "has_failed_trajectory": failed_trajectory is not None,
                "evaluation_available": evaluation is not None
            }
        )
        
        return guidance
        
    def _identify_common_errors(self, failed_trajectory: Optional[Trajectory], evaluation: Optional[TrajectoryEvaluation]) -> List[str]:
        errors = []
        
        if failed_trajectory:
            for step in failed_trajectory.steps:
                if not step.success and step.error:
                    errors.append(f"Step {step.index}: {step.error}")
                    
        if evaluation:
            harmful_steps = [e for e in evaluation.step_evaluations if e.annotation == CriticLabel.HARMFUL]
            for step_eval in harmful_steps[:3]:
                errors.append(f"Issue at step {step_eval.step_index}: {step_eval.summary}")
                
        return errors[:5]
        
    def _calculate_guidance_confidence(self, descriptions: List[Dict], key_steps: List[Dict], evaluation: Optional[TrajectoryEvaluation]) -> float:
        base_confidence = 0.5
        
        if descriptions:
            base_confidence += 0.2
            
        if key_steps:
            base_confidence += len(key_steps) * 0.05
            
        if evaluation:
            if evaluation.is_correct:
                base_confidence += 0.2
            else:
                base_confidence -= 0.1
                
        return min(max(base_confidence, 0.0), 1.0)
        
    async def generate_batch_guidance(self, queries: List[Query], trajectories: Dict[str, Trajectory], evaluations: Dict[str, TrajectoryEvaluation]) -> Dict[str, QueryGuidance]:
        
        guidance_map = {}
        tasks = []
        
        for query in queries:
            trajectory = trajectories.get(query.id)
            evaluation = evaluations.get(query.id)
            
            task = asyncio.create_task(
                self.generate_guidance(query, trajectory, evaluation)
            )
            tasks.append((query.id, task))
            
        for query_id, task in tasks:
            try:
                guidance = await task
                guidance_map[query_id] = guidance
            except Exception as e:
                logger.error(f"Failed to generate guidance for query {query_id}: {e}")
                
        return guidance_map
        
    def improve_guidance(self, guidance: QueryGuidance, new_trajectory: Trajectory, new_evaluation: TrajectoryEvaluation) -> QueryGuidance:
        
        if new_evaluation.is_correct:
            successful_steps = []
            for i, step in enumerate(new_trajectory.steps):
                if step.success:
                    successful_steps.append({
                        "index": i,
                        "action": step.action,
                        "description": f"Successful {step.action.get('type')} at step {i}",
                        "expected_result": "Success"
                    })
                    
            guidance.steps = successful_steps
            guidance.confidence = min(guidance.confidence + 0.1, 1.0)
            
        else:
            new_errors = []
            for step_eval in new_evaluation.step_evaluations:
                if step_eval.annotation == CriticLabel.HARMFUL:
                    new_errors.append(f"Step {step_eval.step_index}: {step_eval.summary}")
                    
            guidance.common_errors.extend(new_errors)
            guidance.common_errors = list(set(guidance.common_errors))[:10]
            
            for i, step in enumerate(guidance.steps):
                if i < len(new_evaluation.step_evaluations):
                    step_eval = new_evaluation.step_evaluations[i]
                    if step_eval.annotation == CriticLabel.HARMFUL:
                        step["expected_result"] = "Likely to fail - needs adjustment"
                        step["special_instruction"] = "Retry with different approach"
                        
        guidance.metadata["iterations"] = guidance.metadata.get("iterations", 0) + 1
        
        return guidance
        
    def export_guidance(self, guidance: QueryGuidance, output_file: str):
        import json
        
        data = {
            "query_id": guidance.query_id,
            "confidence": guidance.confidence,
            "steps": guidance.steps,
            "key_actions": guidance.key_actions,
            "critical_points": guidance.critical_points,
            "common_errors": guidance.common_errors,
            "metadata": guidance.metadata
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Exported guidance to {output_file}")