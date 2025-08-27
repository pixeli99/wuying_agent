import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from .config import CriticLabel, PipelineConfig
from .trajectory_collector import Trajectory, Step

logger = logging.getLogger(__name__)

@dataclass
class StepEvaluation:
    step_index: int
    analysis: str
    summary: str
    annotation: CriticLabel
    confidence: float
    metadata: Dict[str, Any] = None

@dataclass
class TrajectoryEvaluation:
    trajectory_id: str
    text_channel_result: str
    multimodal_channel_result: str
    consensus_result: str
    is_correct: bool
    confidence: float
    step_evaluations: List[StepEvaluation]
    metadata: Dict[str, Any] = None

class StepLevelCritic:
    def __init__(self, config: PipelineConfig, model_client=None):
        self.config = config
        self.model_client = model_client
        
    async def evaluate_step(self, step: Step, context: Optional[Dict] = None) -> StepEvaluation:
        analysis = await self._analyze_step(step, context)
        
        summary = self._summarize_analysis(analysis)
        
        annotation = self._annotate_step(step, analysis)
        
        confidence = self._calculate_confidence(step, analysis, annotation)
        
        evaluation = StepEvaluation(
            step_index=step.index,
            analysis=analysis,
            summary=summary,
            annotation=annotation,
            confidence=confidence,
            metadata={
                "action_type": step.action.get("type"),
                "execution_time": step.execution_time,
                "has_error": step.error is not None
            }
        )
        
        logger.info(f"Step {step.index} evaluated as {annotation.value} with confidence {confidence:.2f}")
        
        return evaluation
        
    async def _analyze_step(self, step: Step, context: Optional[Dict]) -> str:
        if self.model_client:
            prompt = self._build_analysis_prompt(step, context)
            
            response = await self._call_model_for_analysis(prompt)
            return response
            
        else:
            return self._rule_based_analysis(step)
            
    def _build_analysis_prompt(self, step: Step, context: Optional[Dict]) -> str:
        prompt = f"""
        Analyze this UI interaction step:
        
        Action: {step.action}
        Success: {step.success}
        Error: {step.error}
        
        Pre-state elements: {self._extract_ui_elements(step.pre_state)}
        Post-state elements: {self._extract_ui_elements(step.post_state)}
        
        Context: {context if context else 'No additional context'}
        
        Provide detailed analysis of:
        1. What the action attempted to do
        2. Whether it achieved its intended effect
        3. Any side effects or unexpected changes
        4. The overall impact on task completion
        """
        return prompt
        
    def _extract_ui_elements(self, state: Dict) -> List[str]:
        elements = []
        
        if "ui_hierarchy" in state:
            pass
            
        if "windows" in state:
            for window in state.get("windows", []):
                elements.append(f"Window: {window.get('title', 'Unknown')}")
                
        return elements if elements else ["No UI elements extracted"]
        
    async def _call_model_for_analysis(self, prompt: str) -> str:
        return "The action successfully clicked on the target button, causing navigation to the next screen. The UI transition was smooth and all expected elements loaded correctly."
        
    def _rule_based_analysis(self, step: Step) -> str:
        if step.success:
            return f"Action {step.action.get('type')} executed successfully. The UI responded as expected."
        else:
            return f"Action {step.action.get('type')} failed with error: {step.error}. The UI state may be inconsistent."
            
    def _summarize_analysis(self, analysis: str) -> str:
        words = analysis.split()
        if len(words) <= 30:
            return analysis
            
        return " ".join(words[:30]) + "..."
        
    def _annotate_step(self, step: Step, analysis: str) -> CriticLabel:
        if not step.success:
            if step.error and "critical" in step.error.lower():
                return CriticLabel.HARMFUL
            return CriticLabel.NEUTRAL
            
        analysis_lower = analysis.lower()
        
        positive_indicators = ["success", "correct", "expected", "complete", "achieved"]
        negative_indicators = ["failed", "error", "wrong", "incorrect", "unexpected"]
        
        positive_count = sum(1 for ind in positive_indicators if ind in analysis_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in analysis_lower)
        
        if positive_count > negative_count:
            return CriticLabel.GOOD
        elif negative_count > positive_count:
            return CriticLabel.HARMFUL
        else:
            return CriticLabel.NEUTRAL
            
    def _calculate_confidence(self, step: Step, analysis: str, annotation: CriticLabel) -> float:
        base_confidence = 0.5
        
        if step.success:
            base_confidence += 0.2
            
        if step.execution_time < 1.0:
            base_confidence += 0.1
        elif step.execution_time > 5.0:
            base_confidence -= 0.1
            
        if annotation == CriticLabel.GOOD and step.success:
            base_confidence += 0.15
        elif annotation == CriticLabel.HARMFUL and not step.success:
            base_confidence += 0.15
            
        if len(analysis) > 100:
            base_confidence += 0.05
            
        return min(max(base_confidence, 0.0), 1.0)
        
    async def evaluate_trajectory_steps(self, trajectory: Trajectory) -> List[StepEvaluation]:
        evaluations = []
        
        for i, step in enumerate(trajectory.steps):
            context = {
                "query": trajectory.query.natural_instruction,
                "step_number": i + 1,
                "total_steps": len(trajectory.steps),
                "previous_steps": [s.action for s in trajectory.steps[:i]]
            }
            
            evaluation = await self.evaluate_step(step, context)
            evaluations.append(evaluation)
            
        return evaluations

class TrajectoryLevelCritic:
    def __init__(self, config: PipelineConfig, model_client=None):
        self.config = config
        self.model_client = model_client
        self.step_critic = StepLevelCritic(config, model_client)
        
    async def evaluate_trajectory(self, trajectory: Trajectory) -> TrajectoryEvaluation:
        step_evaluations = await self.step_critic.evaluate_trajectory_steps(trajectory)
        
        text_result = await self._text_channel_evaluation(trajectory, step_evaluations)
        
        multimodal_result = await self._multimodal_channel_evaluation(trajectory, step_evaluations)
        
        consensus_result, is_correct = self._compute_consensus(text_result, multimodal_result)
        
        confidence = self._calculate_trajectory_confidence(
            step_evaluations, 
            text_result, 
            multimodal_result, 
            consensus_result
        )
        
        evaluation = TrajectoryEvaluation(
            trajectory_id=trajectory.id,
            text_channel_result=text_result,
            multimodal_channel_result=multimodal_result,
            consensus_result=consensus_result,
            is_correct=is_correct,
            confidence=confidence,
            step_evaluations=step_evaluations,
            metadata={
                "query_difficulty": trajectory.query.difficulty,
                "trajectory_length": len(trajectory.steps),
                "success_rate": trajectory.success_rate,
                "harmful_steps": sum(1 for e in step_evaluations if e.annotation == CriticLabel.HARMFUL),
                "good_steps": sum(1 for e in step_evaluations if e.annotation == CriticLabel.GOOD)
            }
        )
        
        logger.info(f"Trajectory {trajectory.id} evaluated as {'CORRECT' if is_correct else 'INCORRECT'} with confidence {confidence:.2f}")
        
        return evaluation
        
    async def _text_channel_evaluation(self, trajectory: Trajectory, step_evaluations: List[StepEvaluation]) -> str:
        if self.model_client:
            summaries = [eval.summary for eval in step_evaluations]
            
            prompt = f"""
            Based on these step summaries, evaluate if the trajectory correctly completes the task:
            
            Task: {trajectory.query.natural_instruction}
            
            Step summaries:
            {self._format_summaries(summaries)}
            
            Determine if the task was completed correctly (CORRECT/INCORRECT/PARTIAL).
            """
            
            return await self._call_model_for_text_eval(prompt)
            
        else:
            return self._rule_based_text_evaluation(trajectory, step_evaluations)
            
    def _format_summaries(self, summaries: List[str]) -> str:
        formatted = []
        for i, summary in enumerate(summaries):
            formatted.append(f"{i+1}. {summary}")
        return "\n".join(formatted)
        
    async def _call_model_for_text_eval(self, prompt: str) -> str:
        return "CORRECT"
        
    def _rule_based_text_evaluation(self, trajectory: Trajectory, step_evaluations: List[StepEvaluation]) -> str:
        harmful_count = sum(1 for e in step_evaluations if e.annotation == CriticLabel.HARMFUL)
        good_count = sum(1 for e in step_evaluations if e.annotation == CriticLabel.GOOD)
        
        if harmful_count > len(step_evaluations) * 0.3:
            return "INCORRECT"
        elif good_count > len(step_evaluations) * 0.7:
            return "CORRECT"
        else:
            return "PARTIAL"
            
    async def _multimodal_channel_evaluation(self, trajectory: Trajectory, step_evaluations: List[StepEvaluation]) -> str:
        if self.model_client:
            visual_context = self._prepare_visual_context(trajectory)
            
            prompt = f"""
            Evaluate this trajectory using both visual and textual information:
            
            Task: {trajectory.query.natural_instruction}
            
            Visual progression: {len(visual_context)} screenshots captured
            Step evaluations: {self._summarize_evaluations(step_evaluations)}
            
            Does the visual evidence show successful task completion? (CORRECT/INCORRECT/PARTIAL)
            """
            
            return await self._call_model_for_multimodal_eval(prompt, visual_context)
            
        else:
            return self._rule_based_multimodal_evaluation(trajectory, step_evaluations)
            
    def _prepare_visual_context(self, trajectory: Trajectory) -> List[Dict]:
        visual_context = []
        
        for step in trajectory.steps[::max(1, len(trajectory.steps) // 5)]:
            if step.post_state.get("screenshot"):
                visual_context.append({
                    "step": step.index,
                    "screenshot": step.post_state["screenshot"],
                    "action": step.action.get("type")
                })
                
        return visual_context
        
    def _summarize_evaluations(self, evaluations: List[StepEvaluation]) -> str:
        good = sum(1 for e in evaluations if e.annotation == CriticLabel.GOOD)
        neutral = sum(1 for e in evaluations if e.annotation == CriticLabel.NEUTRAL)
        harmful = sum(1 for e in evaluations if e.annotation == CriticLabel.HARMFUL)
        
        return f"GOOD: {good}, NEUTRAL: {neutral}, HARMFUL: {harmful}"
        
    async def _call_model_for_multimodal_eval(self, prompt: str, visual_context: List[Dict]) -> str:
        return "CORRECT"
        
    def _rule_based_multimodal_evaluation(self, trajectory: Trajectory, step_evaluations: List[StepEvaluation]) -> str:
        if trajectory.status != "completed":
            return "INCORRECT"
            
        if trajectory.success_rate > 0.8:
            return "CORRECT"
        elif trajectory.success_rate > 0.5:
            return "PARTIAL"
        else:
            return "INCORRECT"
            
    def _compute_consensus(self, text_result: str, multimodal_result: str) -> Tuple[str, bool]:
        if text_result == multimodal_result:
            consensus = text_result
            is_correct = (consensus == "CORRECT")
        elif text_result == "CORRECT" or multimodal_result == "CORRECT":
            if "INCORRECT" in [text_result, multimodal_result]:
                consensus = "PARTIAL"
                is_correct = False
            else:
                consensus = "PARTIAL"
                is_correct = False
        else:
            consensus = "INCORRECT"
            is_correct = False
            
        return consensus, is_correct
        
    def _calculate_trajectory_confidence(self, 
                                        step_evaluations: List[StepEvaluation],
                                        text_result: str,
                                        multimodal_result: str,
                                        consensus_result: str) -> float:
        base_confidence = 0.5
        
        avg_step_confidence = sum(e.confidence for e in step_evaluations) / len(step_evaluations) if step_evaluations else 0.5
        base_confidence = avg_step_confidence
        
        if text_result == multimodal_result:
            base_confidence += 0.2
        else:
            base_confidence -= 0.1
            
        if consensus_result == "CORRECT":
            good_ratio = sum(1 for e in step_evaluations if e.annotation == CriticLabel.GOOD) / len(step_evaluations) if step_evaluations else 0
            base_confidence += good_ratio * 0.2
        elif consensus_result == "INCORRECT":
            harmful_ratio = sum(1 for e in step_evaluations if e.annotation == CriticLabel.HARMFUL) / len(step_evaluations) if step_evaluations else 0
            base_confidence -= harmful_ratio * 0.2
            
        return min(max(base_confidence, 0.0), 1.0)
        
    async def evaluate_batch(self, trajectories: List[Trajectory]) -> List[TrajectoryEvaluation]:
        tasks = []
        for trajectory in trajectories:
            task = asyncio.create_task(self.evaluate_trajectory(trajectory))
            tasks.append(task)
            
        evaluations = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_evaluations = []
        for i, eval in enumerate(evaluations):
            if isinstance(eval, Exception):
                logger.error(f"Failed to evaluate trajectory {trajectories[i].id}: {eval}")
            else:
                valid_evaluations.append(eval)
                
        return valid_evaluations
        
    def filter_by_quality(self, evaluations: List[TrajectoryEvaluation], threshold: float = None) -> List[TrajectoryEvaluation]:
        threshold = threshold or self.config.critic_threshold
        
        filtered = [
            eval for eval in evaluations 
            if eval.is_correct and eval.confidence >= threshold
        ]
        
        logger.info(f"Filtered {len(filtered)}/{len(evaluations)} trajectories with threshold {threshold}")
        
        return filtered