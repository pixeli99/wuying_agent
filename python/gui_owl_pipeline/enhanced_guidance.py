"""
Enhanced Guidance Generator implementing GUI-Owl's query-specific guidance generation.
Provides VLM-based action descriptions and quality-controlled guidance for difficult queries.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from .config import PipelineConfig
from .trajectory_collector import Trajectory, Step
from .enhanced_evaluator import EnhancedTrajectoryEvaluation, CriticLabel

logger = logging.getLogger(__name__)

@dataclass
class ActionDescription:
    """VLM-generated description of action execution and results."""
    step_index: int
    action_type: str
    action_params: Dict[str, Any]
    execution_description: str  # What the action does
    expected_result: str  # What should happen
    actual_result: str  # What actually happened
    visual_changes: List[str]  # Key visual changes observed
    confidence: float

@dataclass
class QuerySpecificGuidance:
    """Query-specific guidance for difficult tasks."""
    query_id: str
    query_text: str
    difficulty_analysis: str
    key_steps: List[Dict[str, Any]]  # Critical steps for success
    common_failures: List[str]  # Common failure points
    recovery_strategies: List[str]  # How to recover from failures
    action_descriptions: List[ActionDescription]
    success_criteria: List[str]  # How to verify task completion
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedGuidanceGenerator:
    """
    Generates query-specific guidance for difficult queries using VLM and LLM.
    """
    
    def __init__(self, config: PipelineConfig, model_client=None, vlm_client=None):
        self.config = config
        self.model_client = model_client  # LLM for text reasoning
        self.vlm_client = vlm_client  # VLM for visual analysis
        self.guidance_cache: Dict[str, QuerySpecificGuidance] = {}
    
    async def generate_guidance(self,
                               query_id: str,
                               query_text: str,
                               failed_trajectory: Trajectory,
                               evaluation: EnhancedTrajectoryEvaluation) -> QuerySpecificGuidance:
        """
        Generate comprehensive guidance for a difficult query.
        
        Args:
            query_id: Unique query identifier
            query_text: The task instruction
            failed_trajectory: The failed/incomplete trajectory
            evaluation: The trajectory evaluation results
        """
        
        # Analyze why the query is difficult
        difficulty_analysis = await self._analyze_difficulty(
            query_text, failed_trajectory, evaluation
        )
        
        # Generate action descriptions using VLM
        action_descriptions = await self._generate_action_descriptions(
            failed_trajectory, evaluation
        )
        
        # Verify decision consistency
        validated_descriptions = await self._validate_action_descriptions(
            action_descriptions, failed_trajectory
        )
        
        # Extract key steps and failure points
        key_steps = self._extract_key_steps(
            failed_trajectory, evaluation, validated_descriptions
        )
        
        common_failures = self._identify_common_failures(
            evaluation, validated_descriptions
        )
        
        # Generate recovery strategies
        recovery_strategies = await self._generate_recovery_strategies(
            common_failures, failed_trajectory, evaluation
        )
        
        # Define success criteria
        success_criteria = await self._define_success_criteria(
            query_text, key_steps
        )
        
        # Calculate guidance confidence
        confidence = self._calculate_guidance_confidence(
            validated_descriptions, evaluation
        )
        
        guidance = QuerySpecificGuidance(
            query_id=query_id,
            query_text=query_text,
            difficulty_analysis=difficulty_analysis,
            key_steps=key_steps,
            common_failures=common_failures,
            recovery_strategies=recovery_strategies,
            action_descriptions=validated_descriptions,
            success_criteria=success_criteria,
            confidence=confidence,
            metadata={
                "trajectory_id": failed_trajectory.id,
                "harmful_steps": evaluation.harmful_steps,
                "critical_steps": evaluation.critical_steps,
                "consensus_result": evaluation.consensus_result
            }
        )
        
        # Cache the guidance
        self.guidance_cache[query_id] = guidance
        
        logger.info(f"Generated guidance for query {query_id} with confidence {confidence:.2f}")
        
        return guidance
    
    async def _analyze_difficulty(self,
                                 query_text: str,
                                 trajectory: Trajectory,
                                 evaluation: EnhancedTrajectoryEvaluation) -> str:
        """Analyze why this query is difficult."""
        
        if self.model_client:
            prompt = self._build_difficulty_prompt(query_text, trajectory, evaluation)
            return await self._call_model_for_analysis(prompt)
        else:
            return self._rule_based_difficulty_analysis(trajectory, evaluation)
    
    def _build_difficulty_prompt(self,
                                query_text: str,
                                trajectory: Trajectory,
                                evaluation: EnhancedTrajectoryEvaluation) -> str:
        """Build prompt for difficulty analysis."""
        
        harmful_steps_summary = []
        for step_idx in evaluation.harmful_steps[:5]:
            if step_idx < len(evaluation.step_evaluations):
                step_eval = evaluation.step_evaluations[step_idx]
                harmful_steps_summary.append(
                    f"Step {step_idx + 1}: {step_eval.summary}"
                )
        
        prompt = f"""
        Analyze why this task is difficult based on the failed execution:
        
        Task: {query_text}
        
        Execution Summary:
        - Total steps: {len(trajectory.steps)}
        - Success rate: {trajectory.success_rate:.0%}
        - Consensus result: {evaluation.consensus_result}
        - Harmful steps: {len(evaluation.harmful_steps)}
        
        Key Failures:
        {chr(10).join(harmful_steps_summary)}
        
        Identify:
        1. Primary difficulty factors (e.g., UI complexity, timing, state dependencies)
        2. Specific challenges in this execution
        3. Potential ambiguities in the task description
        
        Provide a concise analysis (100-150 words).
        """
        
        return prompt
    
    def _rule_based_difficulty_analysis(self,
                                       trajectory: Trajectory,
                                       evaluation: EnhancedTrajectoryEvaluation) -> str:
        """Rule-based difficulty analysis."""
        
        analysis_parts = []
        
        # Check failure rate
        if evaluation.harmful_steps:
            harmful_ratio = len(evaluation.harmful_steps) / len(trajectory.steps)
            if harmful_ratio > 0.5:
                analysis_parts.append("High failure rate indicates systematic issues")
            elif harmful_ratio > 0.3:
                analysis_parts.append("Multiple failed steps suggest execution challenges")
        
        # Check trajectory length
        if len(trajectory.steps) > 20:
            analysis_parts.append("Long execution sequence increases complexity")
        elif len(trajectory.steps) < 3:
            analysis_parts.append("Task may require more steps than attempted")
        
        # Check specific failure patterns
        if any(step.error and "timeout" in step.error.lower() 
               for step in trajectory.steps):
            analysis_parts.append("Timing-sensitive operations detected")
        
        if any(step.error and "element not found" in step.error.lower() 
               for step in trajectory.steps):
            analysis_parts.append("UI element targeting issues present")
        
        # Check consensus
        if evaluation.text_channel_result != evaluation.multimodal_channel_result:
            analysis_parts.append("Inconsistent evaluation channels suggest ambiguity")
        
        return ". ".join(analysis_parts) if analysis_parts else "Task complexity requires detailed guidance"
    
    async def _generate_action_descriptions(self,
                                           trajectory: Trajectory,
                                           evaluation: EnhancedTrajectoryEvaluation) -> List[ActionDescription]:
        """Generate VLM-based descriptions for each action."""
        
        descriptions = []
        
        for i, step in enumerate(trajectory.steps):
            # Focus on critical and harmful steps
            if i in evaluation.critical_steps or i in evaluation.harmful_steps:
                description = await self._generate_single_action_description(
                    step, i, trajectory, evaluation
                )
                descriptions.append(description)
        
        return descriptions
    
    async def _generate_single_action_description(self,
                                                 step: Step,
                                                 step_index: int,
                                                 trajectory: Trajectory,
                                                 evaluation: EnhancedTrajectoryEvaluation) -> ActionDescription:
        """Generate description for a single action using VLM."""
        
        if self.vlm_client and step.pre_state.get("screenshot") and step.post_state.get("screenshot"):
            # Use VLM to analyze visual changes
            prompt = self._build_vlm_prompt(step, trajectory.query.natural_instruction)
            visual_analysis = await self._call_vlm_for_description(
                prompt,
                step.pre_state["screenshot"],
                step.post_state["screenshot"]
            )
            
            execution_desc = visual_analysis.get("execution", "Action executed")
            expected_result = visual_analysis.get("expected", "State change expected")
            actual_result = visual_analysis.get("actual", "State changed")
            visual_changes = visual_analysis.get("changes", [])
            
        else:
            # Rule-based description
            execution_desc = self._describe_action_execution(step)
            expected_result = self._describe_expected_result(step, trajectory.query.natural_instruction)
            actual_result = self._describe_actual_result(step)
            visual_changes = self._extract_visual_changes(step)
        
        # Calculate confidence based on step evaluation
        step_eval = evaluation.step_evaluations[step_index] if step_index < len(evaluation.step_evaluations) else None
        confidence = step_eval.confidence if step_eval else 0.5
        
        return ActionDescription(
            step_index=step_index,
            action_type=step.action.get("type", "unknown"),
            action_params=step.action.get("params", {}),
            execution_description=execution_desc,
            expected_result=expected_result,
            actual_result=actual_result,
            visual_changes=visual_changes,
            confidence=confidence
        )
    
    def _build_vlm_prompt(self, step: Step, task: str) -> str:
        """Build prompt for VLM analysis."""
        
        return f"""
        Analyze this UI interaction step:
        
        Task: {task}
        Action: {step.action.get('type')} with params {step.action.get('params')}
        
        Compare the before and after screenshots to describe:
        1. What the action attempts to do (execution)
        2. What should happen for task progress (expected)
        3. What actually happened (actual)
        4. Key visual changes observed
        
        Be specific about UI elements and state changes.
        """
    
    def _describe_action_execution(self, step: Step) -> str:
        """Generate rule-based action execution description."""
        
        action_type = step.action.get("type", "unknown")
        params = step.action.get("params", {})
        
        if action_type == "click":
            element = params.get("element", "element")
            coords = params.get("coordinates", "")
            return f"Click on {element} at {coords}" if coords else f"Click on {element}"
        elif action_type == "type":
            text = params.get("text", "")[:30]
            return f"Type '{text}' into input field"
        elif action_type == "scroll":
            direction = params.get("direction", "down")
            amount = params.get("amount", "")
            return f"Scroll {direction} {amount}".strip()
        elif action_type == "swipe":
            direction = params.get("direction", "")
            return f"Swipe {direction}"
        else:
            return f"Execute {action_type} action"
    
    def _describe_expected_result(self, step: Step, task: str) -> str:
        """Describe expected result based on action and task context."""
        
        action_type = step.action.get("type", "")
        
        # Task-specific expectations
        task_lower = task.lower()
        if "login" in task_lower and action_type == "click":
            return "Navigate to login screen or submit credentials"
        elif "search" in task_lower and action_type == "type":
            return "Enter search query in search field"
        elif "navigate" in task_lower and action_type == "click":
            return "Load target page or screen"
        
        # Generic expectations
        if action_type == "click":
            return "Trigger button action or navigation"
        elif action_type == "type":
            return "Input text appears in field"
        elif action_type == "scroll":
            return "Reveal additional content"
        else:
            return "UI state should change appropriately"
    
    def _describe_actual_result(self, step: Step) -> str:
        """Describe actual result based on step outcome."""
        
        if step.success:
            if step.post_state != step.pre_state:
                return "Action completed with visible state change"
            else:
                return "Action completed but no visible change detected"
        else:
            if step.error:
                return f"Action failed: {step.error[:100]}"
            else:
                return "Action failed without error message"
    
    def _extract_visual_changes(self, step: Step) -> List[str]:
        """Extract visual changes between states."""
        
        changes = []
        
        # Compare window titles
        pre_windows = step.pre_state.get("windows", [])
        post_windows = step.post_state.get("windows", [])
        
        if len(pre_windows) != len(post_windows):
            changes.append(f"Window count changed: {len(pre_windows)} → {len(post_windows)}")
        
        # Check for new elements
        if "ui_hierarchy" in step.post_state and "ui_hierarchy" in step.pre_state:
            pre_count = step.pre_state["ui_hierarchy"].get("element_count", 0)
            post_count = step.post_state["ui_hierarchy"].get("element_count", 0)
            if pre_count != post_count:
                changes.append(f"UI elements changed: {pre_count} → {post_count}")
        
        # Check execution time as indicator of loading
        if step.execution_time > 2.0:
            changes.append("Slow transition suggests loading or processing")
        
        return changes if changes else ["No significant visual changes detected"]
    
    async def _validate_action_descriptions(self,
                                           descriptions: List[ActionDescription],
                                           trajectory: Trajectory) -> List[ActionDescription]:
        """
        Validate VLM descriptions against actual execution results.
        Quality control step from GUI-Owl.
        """
        
        validated = []
        
        for desc in descriptions:
            step = trajectory.steps[desc.step_index]
            
            # Check consistency between description and execution
            is_consistent = await self._check_description_consistency(desc, step)
            
            if is_consistent:
                validated.append(desc)
            else:
                # Adjust description if inconsistent
                adjusted = self._adjust_description(desc, step)
                validated.append(adjusted)
                logger.debug(f"Adjusted description for step {desc.step_index}")
        
        return validated
    
    async def _check_description_consistency(self,
                                            description: ActionDescription,
                                            step: Step) -> bool:
        """Check if description matches actual execution."""
        
        # Simple consistency checks
        if step.success and "failed" in description.actual_result.lower():
            return False
        
        if not step.success and "completed" in description.actual_result.lower():
            return False
        
        # Check action type match
        if description.action_type != step.action.get("type", ""):
            return False
        
        return True
    
    def _adjust_description(self,
                           description: ActionDescription,
                           step: Step) -> ActionDescription:
        """Adjust description to match actual execution."""
        
        # Create adjusted description
        adjusted = ActionDescription(
            step_index=description.step_index,
            action_type=step.action.get("type", "unknown"),
            action_params=step.action.get("params", {}),
            execution_description=description.execution_description,
            expected_result=description.expected_result,
            actual_result=self._describe_actual_result(step),
            visual_changes=description.visual_changes,
            confidence=description.confidence * 0.8  # Reduce confidence for adjusted
        )
        
        return adjusted
    
    def _extract_key_steps(self,
                          trajectory: Trajectory,
                          evaluation: EnhancedTrajectoryEvaluation,
                          descriptions: List[ActionDescription]) -> List[Dict[str, Any]]:
        """Extract and summarize key steps for task completion."""
        
        key_steps = []
        
        # Include critical steps identified by evaluator
        for idx in evaluation.critical_steps:
            if idx < len(trajectory.steps):
                step = trajectory.steps[idx]
                step_eval = evaluation.step_evaluations[idx]
                
                # Find corresponding description
                desc = next((d for d in descriptions if d.step_index == idx), None)
                
                key_step = {
                    "index": idx,
                    "action": step.action.get("type"),
                    "target": step.action.get("params", {}).get("element", "unknown"),
                    "purpose": desc.execution_description if desc else step_eval.summary,
                    "critical_because": self._explain_criticality(idx, evaluation, trajectory),
                    "success_indicator": desc.expected_result if desc else "Action completes successfully"
                }
                
                key_steps.append(key_step)
        
        # Add recovery steps (steps that fixed previous errors)
        for i, eval in enumerate(evaluation.step_evaluations):
            if i > 0 and i not in [ks["index"] for ks in key_steps]:
                if (evaluation.step_evaluations[i-1].annotation == CriticLabel.HARMFUL and 
                    eval.annotation == CriticLabel.GOOD):
                    
                    step = trajectory.steps[i]
                    key_steps.append({
                        "index": i,
                        "action": step.action.get("type"),
                        "target": step.action.get("params", {}).get("element", "unknown"),
                        "purpose": f"Recovery from previous error",
                        "critical_because": "Corrects previous failure",
                        "success_indicator": "Previous error resolved"
                    })
        
        # Sort by index
        key_steps.sort(key=lambda x: x["index"])
        
        return key_steps
    
    def _explain_criticality(self,
                            step_index: int,
                            evaluation: EnhancedTrajectoryEvaluation,
                            trajectory: Trajectory) -> str:
        """Explain why a step is critical."""
        
        step = trajectory.steps[step_index]
        action_type = step.action.get("type", "").lower()
        
        # Check if it's a final step
        if step_index == len(trajectory.steps) - 1:
            return "Final step in trajectory"
        
        # Check for specific critical actions
        critical_actions = ["submit", "confirm", "save", "complete", "login"]
        for critical in critical_actions:
            if critical in action_type:
                return f"Performs {critical} operation"
        
        # Check if it's a recovery step
        if step_index > 0:
            if evaluation.step_evaluations[step_index - 1].annotation == CriticLabel.HARMFUL:
                return "Recovers from previous failure"
        
        return "Key navigation or state change"
    
    def _identify_common_failures(self,
                                 evaluation: EnhancedTrajectoryEvaluation,
                                 descriptions: List[ActionDescription]) -> List[str]:
        """Identify common failure patterns."""
        
        failures = []
        
        # Analyze harmful steps
        for idx in evaluation.harmful_steps:
            if idx < len(evaluation.step_evaluations):
                step_eval = evaluation.step_evaluations[idx]
                
                # Extract failure pattern
                if "timeout" in step_eval.analysis.lower():
                    if "Timing-sensitive operation failed" not in failures:
                        failures.append("Timing-sensitive operation failed")
                elif "element not found" in step_eval.analysis.lower():
                    if "UI element targeting failed" not in failures:
                        failures.append("UI element targeting failed")
                elif "no change" in step_eval.analysis.lower():
                    if "Action had no effect" not in failures:
                        failures.append("Action had no effect")
                elif "wrong" in step_eval.analysis.lower() or "incorrect" in step_eval.analysis.lower():
                    if "Incorrect element or action selected" not in failures:
                        failures.append("Incorrect element or action selected")
        
        # Check description inconsistencies
        for desc in descriptions:
            if desc.expected_result != desc.actual_result:
                if desc.confidence < 0.5:
                    if "Unexpected UI behavior" not in failures:
                        failures.append("Unexpected UI behavior")
        
        # Add generic failure if none found
        if not failures:
            failures.append("Task execution did not achieve expected outcome")
        
        return failures
    
    async def _generate_recovery_strategies(self,
                                           common_failures: List[str],
                                           trajectory: Trajectory,
                                           evaluation: EnhancedTrajectoryEvaluation) -> List[str]:
        """Generate strategies to recover from common failures."""
        
        if self.model_client:
            prompt = self._build_recovery_prompt(common_failures, trajectory, evaluation)
            strategies_text = await self._call_model_for_recovery(prompt)
            return strategies_text.split("\n")[:5]  # Limit to 5 strategies
        else:
            return self._rule_based_recovery_strategies(common_failures)
    
    def _build_recovery_prompt(self,
                              failures: List[str],
                              trajectory: Trajectory,
                              evaluation: EnhancedTrajectoryEvaluation) -> str:
        """Build prompt for recovery strategy generation."""
        
        return f"""
        Generate recovery strategies for these common failures:
        
        Task: {trajectory.query.natural_instruction}
        
        Common Failures:
        {chr(10).join(f"- {f}" for f in failures)}
        
        Failed Steps Summary:
        {chr(10).join(f"Step {i+1}: {evaluation.step_evaluations[i].summary}" 
                     for i in evaluation.harmful_steps[:3])}
        
        Provide 3-5 specific recovery strategies that could help complete this task.
        Each strategy should be actionable and specific to the failures observed.
        """
    
    def _rule_based_recovery_strategies(self, failures: List[str]) -> List[str]:
        """Generate rule-based recovery strategies."""
        
        strategies = []
        
        for failure in failures:
            if "timing" in failure.lower():
                strategies.append("Add explicit waits after navigation actions")
                strategies.append("Check for loading indicators before proceeding")
            elif "element not found" in failure.lower():
                strategies.append("Use more specific element selectors")
                strategies.append("Verify page load completion before interaction")
                strategies.append("Try alternative navigation paths")
            elif "no effect" in failure.lower():
                strategies.append("Verify element is interactive before clicking")
                strategies.append("Check if JavaScript needs to load")
            elif "incorrect" in failure.lower():
                strategies.append("Double-check element identification")
                strategies.append("Confirm current page/state before action")
            elif "unexpected" in failure.lower():
                strategies.append("Handle dynamic UI elements")
                strategies.append("Account for varying response times")
        
        # Add generic strategies if needed
        if len(strategies) < 3:
            strategies.extend([
                "Break complex actions into smaller steps",
                "Add verification after each critical action",
                "Consider alternative interaction methods"
            ])
        
        return strategies[:5]
    
    async def _define_success_criteria(self,
                                      query_text: str,
                                      key_steps: List[Dict[str, Any]]) -> List[str]:
        """Define criteria to verify task completion."""
        
        if self.model_client:
            prompt = self._build_success_criteria_prompt(query_text, key_steps)
            criteria_text = await self._call_model_for_criteria(prompt)
            return criteria_text.split("\n")[:5]
        else:
            return self._rule_based_success_criteria(query_text, key_steps)
    
    def _build_success_criteria_prompt(self,
                                      query_text: str,
                                      key_steps: List[Dict[str, Any]]) -> str:
        """Build prompt for success criteria definition."""
        
        return f"""
        Define success criteria for this task:
        
        Task: {query_text}
        
        Key Steps:
        {chr(10).join(f"{s['index']+1}. {s['action']} - {s['purpose']}" for s in key_steps[:5])}
        
        List 3-5 specific, observable criteria that indicate successful task completion.
        Each criterion should be verifiable through UI state or visual evidence.
        """
    
    def _rule_based_success_criteria(self,
                                    query_text: str,
                                    key_steps: List[Dict[str, Any]]) -> List[str]:
        """Generate rule-based success criteria."""
        
        criteria = []
        query_lower = query_text.lower()
        
        # Task-specific criteria
        if "login" in query_lower:
            criteria.extend([
                "User dashboard or home screen is displayed",
                "Login form is no longer visible",
                "User profile or account info is accessible"
            ])
        elif "search" in query_lower:
            criteria.extend([
                "Search results are displayed",
                "Results match the search query",
                "Results count or pagination is visible"
            ])
        elif "create" in query_lower or "add" in query_lower:
            criteria.extend([
                "New item appears in list or view",
                "Success message or confirmation is shown",
                "Form is cleared or reset"
            ])
        elif "delete" in query_lower or "remove" in query_lower:
            criteria.extend([
                "Item no longer appears in list",
                "Deletion confirmation is shown",
                "Count or total is updated"
            ])
        elif "navigate" in query_lower or "go to" in query_lower:
            criteria.extend([
                "Target page or screen is loaded",
                "URL or title matches destination",
                "Expected content is visible"
            ])
        
        # Add key step completion criteria
        for step in key_steps[:2]:
            criteria.append(f"{step['success_indicator']}")
        
        # Generic criteria if needed
        if len(criteria) < 3:
            criteria.extend([
                "All required actions completed without errors",
                "Final UI state matches expected outcome",
                "No error messages or warnings displayed"
            ])
        
        return criteria[:5]
    
    def _calculate_guidance_confidence(self,
                                      descriptions: List[ActionDescription],
                                      evaluation: EnhancedTrajectoryEvaluation) -> float:
        """Calculate confidence in the generated guidance."""
        
        confidence = 0.5  # Base confidence
        
        # Factor in description quality
        if descriptions:
            avg_desc_confidence = sum(d.confidence for d in descriptions) / len(descriptions)
            confidence = avg_desc_confidence
        
        # Adjust based on evaluation consensus
        if evaluation.text_channel_result == evaluation.multimodal_channel_result:
            confidence += 0.1  # Consistent evaluation
        else:
            confidence -= 0.1  # Inconsistent evaluation
        
        # Adjust based on trajectory completion
        if evaluation.consensus_result == "PARTIAL":
            confidence += 0.05  # Some progress made
        elif evaluation.consensus_result == "INCORRECT":
            confidence -= 0.05  # Complete failure
        
        # Adjust based on harmful steps ratio
        if len(evaluation.harmful_steps) > len(evaluation.step_evaluations) * 0.5:
            confidence -= 0.1  # Too many failures
        
        return max(0.1, min(0.95, confidence))
    
    async def generate_batch_guidance(self,
                                     queries: List[Tuple[str, str]],
                                     trajectories: Dict[str, Trajectory],
                                     evaluations: Dict[str, EnhancedTrajectoryEvaluation]) -> Dict[str, QuerySpecificGuidance]:
        """
        Generate guidance for multiple queries in batch.
        
        Args:
            queries: List of (query_id, query_text) tuples
            trajectories: Map of query_id to failed trajectory
            evaluations: Map of query_id to evaluation
        """
        
        tasks = []
        for query_id, query_text in queries:
            if query_id in trajectories and query_id in evaluations:
                task = asyncio.create_task(
                    self.generate_guidance(
                        query_id,
                        query_text,
                        trajectories[query_id],
                        evaluations[query_id]
                    )
                )
                tasks.append((query_id, task))
        
        # Wait for all guidance generation
        guidance_map = {}
        for query_id, task in tasks:
            try:
                guidance = await task
                guidance_map[query_id] = guidance
            except Exception as e:
                logger.error(f"Failed to generate guidance for {query_id}: {e}")
        
        return guidance_map
    
    def apply_guidance(self,
                       query_id: str,
                       current_step: int,
                       current_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply cached guidance to help with current execution.
        
        Returns:
            Suggested action or None if no guidance available
        """
        
        if query_id not in self.guidance_cache:
            return None
        
        guidance = self.guidance_cache[query_id]
        
        # Find relevant key step
        relevant_step = None
        for key_step in guidance.key_steps:
            if key_step["index"] == current_step:
                relevant_step = key_step
                break
        
        if relevant_step:
            # Suggest action based on guidance
            suggestion = {
                "action": relevant_step["action"],
                "target": relevant_step["target"],
                "purpose": relevant_step["purpose"],
                "wait_for": relevant_step["success_indicator"],
                "confidence": guidance.confidence
            }
            
            # Add recovery if previous step failed
            if current_step > 0 and self._should_apply_recovery(current_state):
                suggestion["recovery_strategies"] = guidance.recovery_strategies[:2]
            
            return suggestion
        
        return None
    
    def _should_apply_recovery(self, current_state: Dict[str, Any]) -> bool:
        """Check if recovery strategies should be applied."""
        
        # Check for error indicators
        if current_state.get("error"):
            return True
        
        # Check for stuck state (no recent changes)
        if current_state.get("unchanged_count", 0) > 2:
            return True
        
        return False
    
    async def _call_model_for_analysis(self, prompt: str) -> str:
        """Call LLM for difficulty analysis."""
        # Placeholder for actual model call
        return "Task requires precise element targeting and timing coordination"
    
    async def _call_vlm_for_description(self, prompt: str, pre_screenshot: Any, post_screenshot: Any) -> Dict[str, Any]:
        """Call VLM for visual analysis."""
        # Placeholder for actual VLM call
        return {
            "execution": "Clicked on button",
            "expected": "Navigate to next screen",
            "actual": "Screen transitioned successfully",
            "changes": ["New screen loaded", "Previous elements removed"]
        }
    
    async def _call_model_for_recovery(self, prompt: str) -> str:
        """Call LLM for recovery strategies."""
        # Placeholder
        return "Wait for page load\nVerify element visibility\nUse alternative selectors"
    
    async def _call_model_for_criteria(self, prompt: str) -> str:
        """Call LLM for success criteria."""
        # Placeholder
        return "Target screen is displayed\nExpected elements are visible\nNo error messages shown"
    
    def export_guidance(self, guidance: QuerySpecificGuidance, filepath: str):
        """Export guidance to JSON file."""
        
        data = {
            "query_id": guidance.query_id,
            "query_text": guidance.query_text,
            "difficulty_analysis": guidance.difficulty_analysis,
            "key_steps": guidance.key_steps,
            "common_failures": guidance.common_failures,
            "recovery_strategies": guidance.recovery_strategies,
            "success_criteria": guidance.success_criteria,
            "confidence": guidance.confidence,
            "metadata": guidance.metadata,
            "action_descriptions": [
                {
                    "step_index": desc.step_index,
                    "action_type": desc.action_type,
                    "execution": desc.execution_description,
                    "expected": desc.expected_result,
                    "actual": desc.actual_result,
                    "visual_changes": desc.visual_changes,
                    "confidence": desc.confidence
                }
                for desc in guidance.action_descriptions
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported guidance to {filepath}")