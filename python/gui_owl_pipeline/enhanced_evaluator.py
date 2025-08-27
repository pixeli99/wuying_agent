"""
Enhanced Trajectory Evaluator implementing GUI-Owl's dual-layer critic system.
Provides step-level and trajectory-level evaluation with text and multimodal channels.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .config import CriticLabel, PipelineConfig
from .trajectory_collector import Trajectory, Step

logger = logging.getLogger(__name__)

@dataclass
class EnhancedStepEvaluation:
    """Step-level evaluation with detailed analysis and annotation."""
    step_index: int
    analysis: str  # Detailed explanation of action context and consequences
    summary: str  # 30-word max key insights
    annotation: CriticLabel  # GOOD, NEUTRAL, or HARMFUL
    confidence: float
    action_type: str
    pre_state_summary: str
    post_state_summary: str
    execution_time: float
    metadata: Dict[str, Any] = None

@dataclass
class EnhancedTrajectoryEvaluation:
    """Trajectory-level evaluation with dual-channel consensus mechanism."""
    trajectory_id: str
    text_channel_result: str
    text_channel_reasoning: str
    multimodal_channel_result: str
    multimodal_channel_reasoning: str
    consensus_result: str
    is_correct: bool
    confidence: float
    step_evaluations: List[EnhancedStepEvaluation]
    harmful_steps: List[int]
    critical_steps: List[int]  # Key steps for task completion
    metadata: Dict[str, Any] = None

class EnhancedStepLevelCritic:
    """
    Implements GUI-Owl's step-level critic for fine-grained action evaluation.
    """
    
    def __init__(self, config: PipelineConfig, model_client=None):
        self.config = config
        self.model_client = model_client
        self.max_summary_words = 30
        
    async def evaluate_step(self, 
                           step: Step, 
                           trajectory_context: Dict,
                           previous_evaluations: List[EnhancedStepEvaluation] = None) -> EnhancedStepEvaluation:
        """
        Evaluate a single step with context awareness.
        
        Args:
            step: The step to evaluate
            trajectory_context: Overall trajectory information
            previous_evaluations: Previous step evaluations for context
        """
        # Extract state summaries
        pre_state_summary = self._summarize_state(step.pre_state)
        post_state_summary = self._summarize_state(step.post_state)
        
        # Generate detailed analysis
        analysis = await self._generate_analysis(
            step, 
            pre_state_summary, 
            post_state_summary,
            trajectory_context,
            previous_evaluations
        )
        
        # Create concise summary (max 30 words)
        summary = self._create_summary(analysis)
        
        # Determine annotation based on analysis
        annotation = self._determine_annotation(step, analysis, trajectory_context)
        
        # Calculate confidence
        confidence = self._calculate_step_confidence(
            step, 
            annotation, 
            analysis,
            previous_evaluations
        )
        
        evaluation = EnhancedStepEvaluation(
            step_index=step.index,
            analysis=analysis,
            summary=summary,
            annotation=annotation,
            confidence=confidence,
            action_type=step.action.get("type", "unknown"),
            pre_state_summary=pre_state_summary,
            post_state_summary=post_state_summary,
            execution_time=step.execution_time,
            metadata={
                "has_error": step.error is not None,
                "retry_count": step.metadata.get("retry_count", 0) if step.metadata else 0,
                "is_correction": self._is_correction_step(step, previous_evaluations)
            }
        )
        
        logger.debug(f"Step {step.index} evaluated: {annotation.value} (confidence: {confidence:.2f})")
        
        return evaluation
    
    def _summarize_state(self, state: Dict) -> str:
        """Extract key information from UI state."""
        summary_parts = []
        
        # Extract window/screen information
        if "windows" in state:
            window_titles = [w.get("title", "Unknown") for w in state["windows"][:3]]
            if window_titles:
                summary_parts.append(f"Windows: {', '.join(window_titles)}")
        
        # Extract active element info
        if "active_element" in state:
            elem = state["active_element"]
            summary_parts.append(f"Active: {elem.get('type', 'unknown')} '{elem.get('text', '')[:20]}'")
        
        # Extract key UI elements count
        if "ui_hierarchy" in state and isinstance(state["ui_hierarchy"], dict):
            elem_count = state["ui_hierarchy"].get("element_count", 0)
            summary_parts.append(f"Elements: {elem_count}")
        
        return " | ".join(summary_parts) if summary_parts else "No state information"
    
    async def _generate_analysis(self, 
                                step: Step,
                                pre_state: str,
                                post_state: str,
                                trajectory_context: Dict,
                                previous_evaluations: List[EnhancedStepEvaluation]) -> str:
        """Generate detailed analysis of the step."""
        
        if self.model_client:
            # Use LLM for analysis
            prompt = self._build_analysis_prompt(
                step, pre_state, post_state, 
                trajectory_context, previous_evaluations
            )
            return await self._call_model_for_analysis(prompt)
        else:
            # Rule-based analysis
            return self._rule_based_analysis(
                step, pre_state, post_state, trajectory_context
            )
    
    def _build_analysis_prompt(self, 
                              step: Step,
                              pre_state: str,
                              post_state: str,
                              trajectory_context: Dict,
                              previous_evaluations: List[EnhancedStepEvaluation]) -> str:
        """Build prompt for LLM-based analysis."""
        
        recent_actions = []
        if previous_evaluations:
            recent_actions = [
                f"{i+1}. {eval.action_type}: {eval.summary}"
                for i, eval in enumerate(previous_evaluations[-3:])
            ]
        
        prompt = f"""
        Analyze this UI interaction step in detail:
        
        Task: {trajectory_context.get('task', 'Unknown task')}
        Step {step.index + 1} of {trajectory_context.get('total_steps', 'unknown')}
        
        Previous Actions:
        {chr(10).join(recent_actions) if recent_actions else 'None'}
        
        Current Action:
        - Type: {step.action.get('type')}
        - Parameters: {step.action.get('params')}
        - Success: {step.success}
        - Error: {step.error if step.error else 'None'}
        - Execution Time: {step.execution_time:.2f}s
        
        Pre-State: {pre_state}
        Post-State: {post_state}
        
        Provide analysis covering:
        1. What the action attempted and why (in task context)
        2. Whether it achieved the intended effect
        3. Any side effects or unexpected behaviors
        4. Impact on overall task progress
        5. Whether this action moves closer to task completion
        """
        
        return prompt
    
    def _rule_based_analysis(self, 
                            step: Step,
                            pre_state: str,
                            post_state: str,
                            trajectory_context: Dict) -> str:
        """Generate rule-based analysis when model is unavailable."""
        
        analysis_parts = []
        action_type = step.action.get("type", "unknown")
        
        # Analyze action intent
        if action_type == "click":
            target = step.action.get("params", {}).get("element", "element")
            analysis_parts.append(f"Attempted to click {target}")
        elif action_type == "type":
            text = step.action.get("params", {}).get("text", "")[:50]
            analysis_parts.append(f"Attempted to input text: '{text}'")
        elif action_type == "scroll":
            direction = step.action.get("params", {}).get("direction", "down")
            analysis_parts.append(f"Attempted to scroll {direction}")
        else:
            analysis_parts.append(f"Executed {action_type} action")
        
        # Analyze outcome
        if step.success:
            analysis_parts.append("Action completed successfully")
            if pre_state != post_state:
                analysis_parts.append("UI state changed as expected")
            else:
                analysis_parts.append("Warning: No visible state change detected")
        else:
            analysis_parts.append(f"Action failed: {step.error}")
            analysis_parts.append("Task progress may be blocked")
        
        # Analyze execution time
        if step.execution_time > 3.0:
            analysis_parts.append(f"Slow execution ({step.execution_time:.1f}s) may indicate issues")
        
        # Context analysis
        progress = (step.index + 1) / trajectory_context.get('total_steps', 1) * 100
        analysis_parts.append(f"Task progress: {progress:.0f}%")
        
        return ". ".join(analysis_parts)
    
    def _create_summary(self, analysis: str) -> str:
        """Create a concise summary (max 30 words) from the analysis."""
        
        # Extract key phrases
        key_phrases = []
        
        # Look for action outcomes
        if "successfully" in analysis.lower():
            key_phrases.append("successful")
        elif "failed" in analysis.lower():
            key_phrases.append("failed")
        
        # Look for state changes
        if "changed" in analysis.lower():
            key_phrases.append("state changed")
        elif "no change" in analysis.lower() or "no visible" in analysis.lower():
            key_phrases.append("no change")
        
        # Look for progress indicators
        if "progress" in analysis.lower():
            if "blocked" in analysis.lower():
                key_phrases.append("progress blocked")
            elif "toward" in analysis.lower() or "closer" in analysis.lower():
                key_phrases.append("progressing")
        
        # Build summary from analysis
        sentences = analysis.split('. ')
        summary_words = []
        
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in ["attempted", "executed", "action"]):
                words = sentence.split()
                summary_words.extend(words[:10])
                break
        
        # Add outcome
        if key_phrases:
            summary_words.append("-")
            summary_words.extend(key_phrases[:2])
        
        # Ensure max 30 words
        if len(summary_words) > self.max_summary_words:
            summary_words = summary_words[:self.max_summary_words-1]
            summary_words.append("...")
        
        return " ".join(summary_words)
    
    def _determine_annotation(self, 
                            step: Step,
                            analysis: str,
                            trajectory_context: Dict) -> CriticLabel:
        """
        Determine step annotation based on comprehensive analysis.
        """
        
        analysis_lower = analysis.lower()
        
        # Check for harmful indicators
        harmful_indicators = [
            "failed", "error", "blocked", "incorrect", "wrong",
            "crashed", "unresponsive", "timeout", "invalid"
        ]
        
        # Check for good indicators
        good_indicators = [
            "successfully", "completed", "correct", "expected",
            "progressing", "achieved", "validated", "confirmed"
        ]
        
        # Check for neutral indicators
        neutral_indicators = [
            "no change", "waiting", "loading", "preparing",
            "optional", "alternative"
        ]
        
        # Count indicators
        harmful_count = sum(1 for ind in harmful_indicators if ind in analysis_lower)
        good_count = sum(1 for ind in good_indicators if ind in analysis_lower)
        neutral_count = sum(1 for ind in neutral_indicators if ind in analysis_lower)
        
        # Decision logic
        if not step.success or step.error:
            # Failed steps are generally harmful unless they're retries
            if step.metadata and step.metadata.get("is_retry"):
                return CriticLabel.NEUTRAL
            return CriticLabel.HARMFUL
        
        # Weight the counts
        if harmful_count >= 2:
            return CriticLabel.HARMFUL
        elif good_count >= 2:
            return CriticLabel.GOOD
        elif harmful_count > good_count:
            return CriticLabel.HARMFUL
        elif good_count > harmful_count:
            return CriticLabel.GOOD
        else:
            # Check criticality for task
            if self._is_critical_step(step, trajectory_context):
                return CriticLabel.GOOD if step.success else CriticLabel.HARMFUL
            return CriticLabel.NEUTRAL
    
    def _is_critical_step(self, step: Step, trajectory_context: Dict) -> bool:
        """Determine if this step is critical for task completion."""
        
        action_type = step.action.get("type", "")
        
        # Critical action types
        critical_actions = ["submit", "confirm", "save", "complete", "finish"]
        if any(action in action_type.lower() for action in critical_actions):
            return True
        
        # Check if it's a final step
        if step.index == trajectory_context.get('total_steps', 0) - 1:
            return True
        
        # Check task-specific critical points
        task = trajectory_context.get('task', '').lower()
        if "login" in task and action_type == "click":
            return True
        if "payment" in task and action_type in ["type", "submit"]:
            return True
        
        return False
    
    def _is_correction_step(self, 
                           step: Step,
                           previous_evaluations: List[EnhancedStepEvaluation]) -> bool:
        """Check if this step is correcting a previous error."""
        
        if not previous_evaluations:
            return False
        
        # Check if previous step was harmful
        if previous_evaluations[-1].annotation == CriticLabel.HARMFUL:
            # Check if current action targets same element or similar action
            prev_action = previous_evaluations[-1].action_type
            curr_action = step.action.get("type", "")
            
            if prev_action == curr_action:
                return True
            if "retry" in str(step.metadata).lower():
                return True
        
        return False
    
    def _calculate_step_confidence(self,
                                  step: Step,
                                  annotation: CriticLabel,
                                  analysis: str,
                                  previous_evaluations: List[EnhancedStepEvaluation]) -> float:
        """Calculate confidence score for step evaluation."""
        
        confidence = 0.5  # Base confidence
        
        # Success/failure alignment
        if step.success and annotation == CriticLabel.GOOD:
            confidence += 0.2
        elif not step.success and annotation == CriticLabel.HARMFUL:
            confidence += 0.2
        elif step.success and annotation == CriticLabel.HARMFUL:
            confidence -= 0.1  # Contradiction
        
        # Execution time factor
        if step.execution_time < 0.5:
            confidence += 0.1  # Fast execution
        elif step.execution_time > 5.0:
            confidence -= 0.1  # Slow execution
        
        # Analysis depth factor
        if len(analysis) > 200:
            confidence += 0.1  # Detailed analysis
        
        # Historical consistency
        if previous_evaluations:
            recent_annotations = [e.annotation for e in previous_evaluations[-3:]]
            if annotation in recent_annotations:
                confidence += 0.05  # Consistent pattern
        
        # Correction factor
        if step.metadata and step.metadata.get("is_correction"):
            if annotation == CriticLabel.GOOD:
                confidence += 0.15  # Successful correction
        
        return np.clip(confidence, 0.0, 1.0)

class EnhancedTrajectoryLevelCritic:
    """
    Implements GUI-Owl's trajectory-level critic with dual-channel evaluation.
    """
    
    def __init__(self, config: PipelineConfig, model_client=None):
        self.config = config
        self.model_client = model_client
        self.step_critic = EnhancedStepLevelCritic(config, model_client)
    
    async def evaluate_trajectory(self, trajectory: Trajectory) -> EnhancedTrajectoryEvaluation:
        """
        Evaluate entire trajectory using dual-channel consensus mechanism.
        """
        
        # Step-level evaluation
        trajectory_context = {
            "task": trajectory.query.natural_instruction,
            "total_steps": len(trajectory.steps),
            "platform": self.config.platform.value
        }
        
        step_evaluations = []
        for step in trajectory.steps:
            step_eval = await self.step_critic.evaluate_step(
                step,
                trajectory_context,
                step_evaluations
            )
            step_evaluations.append(step_eval)
        
        # Text channel evaluation (based on step summaries)
        text_result, text_reasoning = await self._text_channel_evaluation(
            trajectory, step_evaluations
        )
        
        # Multimodal channel evaluation (combining visual and text)
        multimodal_result, multimodal_reasoning = await self._multimodal_channel_evaluation(
            trajectory, step_evaluations
        )
        
        # Consensus mechanism
        consensus_result, is_correct = self._compute_consensus(
            text_result, multimodal_result
        )
        
        # Identify harmful and critical steps
        harmful_steps = [
            eval.step_index 
            for eval in step_evaluations 
            if eval.annotation == CriticLabel.HARMFUL
        ]
        
        critical_steps = self._identify_critical_steps(trajectory, step_evaluations)
        
        # Calculate overall confidence
        confidence = self._calculate_trajectory_confidence(
            step_evaluations,
            text_result,
            multimodal_result,
            consensus_result,
            harmful_steps
        )
        
        evaluation = EnhancedTrajectoryEvaluation(
            trajectory_id=trajectory.id,
            text_channel_result=text_result,
            text_channel_reasoning=text_reasoning,
            multimodal_channel_result=multimodal_result,
            multimodal_channel_reasoning=multimodal_reasoning,
            consensus_result=consensus_result,
            is_correct=is_correct,
            confidence=confidence,
            step_evaluations=step_evaluations,
            harmful_steps=harmful_steps,
            critical_steps=critical_steps,
            metadata={
                "query_difficulty": trajectory.query.difficulty,
                "trajectory_length": len(trajectory.steps),
                "execution_time": sum(s.execution_time for s in trajectory.steps),
                "success_rate": trajectory.success_rate,
                "good_steps": sum(1 for e in step_evaluations if e.annotation == CriticLabel.GOOD),
                "neutral_steps": sum(1 for e in step_evaluations if e.annotation == CriticLabel.NEUTRAL)
            }
        )
        
        logger.info(
            f"Trajectory {trajectory.id} evaluated: "
            f"Text={text_result}, Multimodal={multimodal_result}, "
            f"Consensus={consensus_result} ({'CORRECT' if is_correct else 'INCORRECT'})"
        )
        
        return evaluation
    
    async def _text_channel_evaluation(self, 
                                      trajectory: Trajectory,
                                      step_evaluations: List[EnhancedStepEvaluation]) -> Tuple[str, str]:
        """
        Text reasoning channel based on step summaries.
        """
        
        if self.model_client:
            # Use LLM for semantic reasoning
            prompt = self._build_text_eval_prompt(trajectory, step_evaluations)
            result = await self._call_model_for_text_eval(prompt)
            reasoning = f"Based on step progression and outcomes"
        else:
            # Rule-based evaluation
            result, reasoning = self._rule_based_text_evaluation(
                trajectory, step_evaluations
            )
        
        return result, reasoning
    
    def _build_text_eval_prompt(self, 
                               trajectory: Trajectory,
                               step_evaluations: List[EnhancedStepEvaluation]) -> str:
        """Build prompt for text channel evaluation."""
        
        step_summaries = []
        for eval in step_evaluations:
            status = "✓" if eval.annotation == CriticLabel.GOOD else "✗" if eval.annotation == CriticLabel.HARMFUL else "○"
            step_summaries.append(f"{status} Step {eval.step_index + 1}: {eval.summary}")
        
        prompt = f"""
        Evaluate if this task was completed correctly based on step summaries:
        
        Task: {trajectory.query.natural_instruction}
        
        Execution Summary:
        {chr(10).join(step_summaries)}
        
        Statistics:
        - Total steps: {len(step_evaluations)}
        - Successful steps: {sum(1 for e in step_evaluations if e.annotation == CriticLabel.GOOD)}
        - Failed steps: {sum(1 for e in step_evaluations if e.annotation == CriticLabel.HARMFUL)}
        - Neutral steps: {sum(1 for e in step_evaluations if e.annotation == CriticLabel.NEUTRAL)}
        
        Based on the step progression and outcomes, determine:
        1. Was the task objective achieved?
        2. Were there critical failures that prevent task completion?
        3. Overall assessment: CORRECT, INCORRECT, or PARTIAL
        
        Provide your assessment.
        """
        
        return prompt
    
    def _rule_based_text_evaluation(self,
                                   trajectory: Trajectory,
                                   step_evaluations: List[EnhancedStepEvaluation]) -> Tuple[str, str]:
        """Rule-based text channel evaluation."""
        
        # Count step annotations
        good_count = sum(1 for e in step_evaluations if e.annotation == CriticLabel.GOOD)
        harmful_count = sum(1 for e in step_evaluations if e.annotation == CriticLabel.HARMFUL)
        neutral_count = sum(1 for e in step_evaluations if e.annotation == CriticLabel.NEUTRAL)
        total_steps = len(step_evaluations)
        
        # Check critical steps
        critical_success = all(
            step_evaluations[idx].annotation != CriticLabel.HARMFUL
            for idx in self._identify_critical_steps(trajectory, step_evaluations)
        )
        
        # Decision logic
        good_ratio = good_count / total_steps if total_steps > 0 else 0
        harmful_ratio = harmful_count / total_steps if total_steps > 0 else 0
        
        reasoning_parts = []
        
        if not critical_success:
            result = "INCORRECT"
            reasoning_parts.append("Critical steps failed")
        elif harmful_ratio > 0.3:
            result = "INCORRECT"
            reasoning_parts.append(f"Too many harmful steps ({harmful_count}/{total_steps})")
        elif good_ratio > 0.7:
            result = "CORRECT"
            reasoning_parts.append(f"Majority of steps successful ({good_count}/{total_steps})")
        elif good_ratio > 0.5:
            result = "PARTIAL"
            reasoning_parts.append(f"Moderate success rate ({good_count}/{total_steps})")
        else:
            result = "INCORRECT"
            reasoning_parts.append(f"Insufficient successful steps ({good_count}/{total_steps})")
        
        # Add trajectory completion status
        if trajectory.status == "completed":
            reasoning_parts.append("Trajectory completed")
        else:
            reasoning_parts.append(f"Trajectory status: {trajectory.status}")
        
        reasoning = ". ".join(reasoning_parts)
        
        return result, reasoning
    
    async def _multimodal_channel_evaluation(self,
                                            trajectory: Trajectory,
                                            step_evaluations: List[EnhancedStepEvaluation]) -> Tuple[str, str]:
        """
        Multimodal reasoning channel combining visual and text information.
        """
        
        if self.model_client:
            # Use VLM for multimodal evaluation
            visual_context = self._prepare_visual_context(trajectory, step_evaluations)
            prompt = self._build_multimodal_eval_prompt(trajectory, step_evaluations, visual_context)
            result = await self._call_model_for_multimodal_eval(prompt, visual_context)
            reasoning = "Visual and textual evidence analyzed"
        else:
            # Rule-based evaluation
            result, reasoning = self._rule_based_multimodal_evaluation(
                trajectory, step_evaluations
            )
        
        return result, reasoning
    
    def _prepare_visual_context(self, 
                               trajectory: Trajectory,
                               step_evaluations: List[EnhancedStepEvaluation]) -> List[Dict]:
        """Prepare visual context for multimodal evaluation."""
        
        visual_context = []
        
        # Include critical steps and evenly sampled steps
        critical_steps = self._identify_critical_steps(trajectory, step_evaluations)
        sample_interval = max(1, len(trajectory.steps) // 5)
        
        for i, step in enumerate(trajectory.steps):
            if i in critical_steps or i % sample_interval == 0 or i == len(trajectory.steps) - 1:
                if step.post_state.get("screenshot"):
                    visual_context.append({
                        "step_index": i,
                        "screenshot": step.post_state["screenshot"],
                        "action": step.action,
                        "evaluation": step_evaluations[i].annotation.value,
                        "is_critical": i in critical_steps
                    })
        
        return visual_context
    
    def _build_multimodal_eval_prompt(self,
                                     trajectory: Trajectory,
                                     step_evaluations: List[EnhancedStepEvaluation],
                                     visual_context: List[Dict]) -> str:
        """Build prompt for multimodal evaluation."""
        
        prompt = f"""
        Evaluate task completion using visual and textual evidence:
        
        Task: {trajectory.query.natural_instruction}
        
        Visual Evidence:
        - {len(visual_context)} key screenshots captured
        - Critical steps: {[vc['step_index'] for vc in visual_context if vc['is_critical']]}
        
        Step Outcomes:
        - Successful: {sum(1 for e in step_evaluations if e.annotation == CriticLabel.GOOD)}
        - Failed: {sum(1 for e in step_evaluations if e.annotation == CriticLabel.HARMFUL)}
        - Total: {len(step_evaluations)}
        
        Analyze the visual progression and determine if:
        1. The UI states show progress toward the goal
        2. The final state matches expected completion
        3. Visual evidence confirms successful task execution
        
        Assessment: CORRECT, INCORRECT, or PARTIAL
        """
        
        return prompt
    
    def _rule_based_multimodal_evaluation(self,
                                         trajectory: Trajectory,
                                         step_evaluations: List[EnhancedStepEvaluation]) -> Tuple[str, str]:
        """Rule-based multimodal evaluation."""
        
        reasoning_parts = []
        
        # Check trajectory completion
        if trajectory.status != "completed":
            result = "INCORRECT"
            reasoning_parts.append(f"Trajectory not completed (status: {trajectory.status})")
        else:
            # Check visual evidence availability
            screenshots_available = sum(
                1 for step in trajectory.steps 
                if step.post_state.get("screenshot")
            )
            
            if screenshots_available < len(trajectory.steps) * 0.5:
                # Limited visual evidence, rely more on step evaluations
                good_ratio = sum(
                    1 for e in step_evaluations 
                    if e.annotation == CriticLabel.GOOD
                ) / len(step_evaluations)
                
                if good_ratio > 0.7:
                    result = "CORRECT"
                    reasoning_parts.append("Limited visual but strong step performance")
                else:
                    result = "PARTIAL"
                    reasoning_parts.append("Limited visual evidence for verification")
            else:
                # Good visual coverage
                if trajectory.success_rate > 0.8:
                    result = "CORRECT"
                    reasoning_parts.append(f"High success rate ({trajectory.success_rate:.0%})")
                elif trajectory.success_rate > 0.5:
                    result = "PARTIAL"
                    reasoning_parts.append(f"Moderate success rate ({trajectory.success_rate:.0%})")
                else:
                    result = "INCORRECT"
                    reasoning_parts.append(f"Low success rate ({trajectory.success_rate:.0%})")
            
            reasoning_parts.append(f"Visual evidence: {screenshots_available}/{len(trajectory.steps)} steps")
        
        reasoning = ". ".join(reasoning_parts)
        
        return result, reasoning
    
    def _compute_consensus(self, text_result: str, multimodal_result: str) -> Tuple[str, bool]:
        """
        Compute consensus between text and multimodal channels.
        Following GUI-Owl: trajectory is correct only if both channels agree on CORRECT.
        """
        
        if text_result == "CORRECT" and multimodal_result == "CORRECT":
            # Both channels agree on success
            consensus = "CORRECT"
            is_correct = True
        elif text_result == multimodal_result:
            # Channels agree but not on CORRECT
            consensus = text_result
            is_correct = False
        elif "INCORRECT" in [text_result, multimodal_result]:
            # At least one channel says incorrect
            consensus = "INCORRECT"
            is_correct = False
        else:
            # Mixed results (e.g., CORRECT and PARTIAL)
            consensus = "PARTIAL"
            is_correct = False
        
        return consensus, is_correct
    
    def _identify_critical_steps(self,
                                trajectory: Trajectory,
                                step_evaluations: List[EnhancedStepEvaluation]) -> List[int]:
        """Identify critical steps for task completion."""
        
        critical_indices = []
        
        # Last step is usually critical
        if len(trajectory.steps) > 0:
            critical_indices.append(len(trajectory.steps) - 1)
        
        # Steps with specific action types
        critical_actions = ["submit", "confirm", "save", "complete", "login", "pay"]
        for i, step in enumerate(trajectory.steps):
            action_type = step.action.get("type", "").lower()
            if any(critical in action_type for critical in critical_actions):
                critical_indices.append(i)
        
        # Steps that recovered from errors
        for i, eval in enumerate(step_evaluations):
            if i > 0 and step_evaluations[i-1].annotation == CriticLabel.HARMFUL:
                if eval.annotation == CriticLabel.GOOD:
                    critical_indices.append(i)
        
        # Steps with high confidence good evaluations
        for i, eval in enumerate(step_evaluations):
            if eval.annotation == CriticLabel.GOOD and eval.confidence > 0.8:
                critical_indices.append(i)
        
        return sorted(list(set(critical_indices)))
    
    def _calculate_trajectory_confidence(self,
                                        step_evaluations: List[EnhancedStepEvaluation],
                                        text_result: str,
                                        multimodal_result: str,
                                        consensus_result: str,
                                        harmful_steps: List[int]) -> float:
        """Calculate overall trajectory confidence."""
        
        # Start with average step confidence
        if step_evaluations:
            avg_confidence = np.mean([e.confidence for e in step_evaluations])
        else:
            avg_confidence = 0.5
        
        confidence = avg_confidence
        
        # Channel agreement bonus
        if text_result == multimodal_result:
            confidence += 0.2
        else:
            confidence -= 0.1
        
        # Consensus result adjustment
        if consensus_result == "CORRECT":
            confidence += 0.15
        elif consensus_result == "INCORRECT":
            confidence -= 0.15
        
        # Harmful steps penalty
        if harmful_steps:
            harmful_ratio = len(harmful_steps) / len(step_evaluations)
            confidence -= harmful_ratio * 0.2
        
        # Critical steps success bonus
        critical_steps = self._identify_critical_steps(
            None, step_evaluations  # trajectory not needed for this
        )
        if critical_steps:
            critical_success = all(
                step_evaluations[idx].annotation != CriticLabel.HARMFUL
                for idx in critical_steps
                if idx < len(step_evaluations)
            )
            if critical_success:
                confidence += 0.1
        
        return np.clip(confidence, 0.0, 1.0)
    
    async def _call_model_for_analysis(self, prompt: str) -> str:
        """Call LLM for step analysis."""
        # Placeholder for actual model call
        return "Step executed successfully with expected state transition."
    
    async def _call_model_for_text_eval(self, prompt: str) -> str:
        """Call LLM for text channel evaluation."""
        # Placeholder for actual model call
        return "CORRECT"
    
    async def _call_model_for_multimodal_eval(self, prompt: str, visual_context: List[Dict]) -> str:
        """Call VLM for multimodal evaluation."""
        # Placeholder for actual model call
        return "CORRECT"