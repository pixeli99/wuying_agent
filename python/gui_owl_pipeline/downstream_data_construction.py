"""
Downstream Data Construction Pipeline implementing GUI-Owl's data synthesis strategies.
Generates grounding, task planning, and action semantic data from collected trajectories.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import numpy as np
from collections import defaultdict
from .config import PipelineConfig
from .trajectory_collector import Trajectory, Step
from .enhanced_evaluator import EnhancedTrajectoryEvaluation, CriticLabel

logger = logging.getLogger(__name__)

@dataclass
class GroundingData:
    """UI element grounding data."""
    screenshot: Any  # Screenshot data
    ui_elements: List[Dict[str, Any]]  # UI element descriptions and locations
    functional_descriptions: List[str]  # What each element does
    layout_descriptions: List[str]  # Spatial relationships
    fine_grained_text: List[Dict[str, Any]]  # Character/word-level text locations
    source: str  # Data source (trajectory, a11y, synthesis)
    confidence: float

@dataclass
class TaskPlanningData:
    """Task planning and execution data."""
    task_description: str
    execution_plan: List[Dict[str, Any]]  # Step-by-step plan
    key_decisions: List[Dict[str, Any]]  # Critical decision points
    preconditions: List[str]  # What must be true before execution
    postconditions: List[str]  # What should be true after execution
    alternative_paths: List[List[Dict[str, Any]]]  # Alternative execution paths
    source: str  # distilled from trajectory or LLM
    confidence: float

@dataclass
class ActionSemanticData:
    """Action semantic understanding data."""
    pre_screenshot: Any
    post_screenshot: Any
    action: Dict[str, Any]
    action_description: str  # Natural language description
    expected_effect: str  # What should happen
    actual_effect: str  # What actually happened
    semantic_category: str  # navigation, input, selection, etc.
    confidence: float

@dataclass
class ReasoningData:
    """Chain-of-thought reasoning data."""
    task: str
    reasoning_steps: List[str]  # Step-by-step reasoning
    action_justifications: List[str]  # Why each action was chosen
    state_interpretations: List[str]  # Understanding of current state
    goal_tracking: List[str]  # Progress toward goal
    style: str  # hint style used (concise, detailed, etc.)
    source: str  # offline synthesis or online generation

class DownstreamDataConstructor:
    """
    Constructs various types of downstream training data from trajectories.
    """
    
    def __init__(self, config: PipelineConfig, model_client=None):
        self.config = config
        self.model_client = model_client
        self.data_cache = {
            "grounding": [],
            "planning": [],
            "semantics": [],
            "reasoning": []
        }
    
    async def construct_all_data(self,
                                trajectories: List[Trajectory],
                                evaluations: List[EnhancedTrajectoryEvaluation]) -> Dict[str, List]:
        """
        Construct all types of downstream data from trajectories.
        """
        
        # Filter high-quality trajectories
        quality_pairs = [(t, e) for t, e in zip(trajectories, evaluations) 
                         if e.is_correct or e.consensus_result == "PARTIAL"]
        
        logger.info(f"Constructing downstream data from {len(quality_pairs)} quality trajectories")
        
        # Construct different data types
        grounding_data = await self.construct_grounding_data(quality_pairs)
        planning_data = await self.construct_planning_data(quality_pairs)
        semantic_data = await self.construct_action_semantic_data(quality_pairs)
        reasoning_data = await self.construct_reasoning_data(quality_pairs)
        
        # Update cache
        self.data_cache["grounding"].extend(grounding_data)
        self.data_cache["planning"].extend(planning_data)
        self.data_cache["semantics"].extend(semantic_data)
        self.data_cache["reasoning"].extend(reasoning_data)
        
        return {
            "grounding": grounding_data,
            "planning": planning_data,
            "semantics": semantic_data,
            "reasoning": reasoning_data
        }
    
    async def construct_grounding_data(self,
                                      trajectory_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]]) -> List[GroundingData]:
        """
        Construct UI element grounding data.
        Includes element localization, functional descriptions, and layout understanding.
        """
        
        grounding_data = []
        
        for trajectory, evaluation in trajectory_pairs:
            # Process each step with good evaluation
            for i, (step, step_eval) in enumerate(zip(trajectory.steps, evaluation.step_evaluations)):
                if step_eval.annotation != CriticLabel.HARMFUL:
                    data = await self._extract_grounding_from_step(
                        step, step_eval, trajectory.query.natural_instruction
                    )
                    if data:
                        grounding_data.append(data)
        
        # Synthesize additional grounding data if model available
        if self.model_client:
            synthetic_data = await self._synthesize_grounding_data(trajectory_pairs)
            grounding_data.extend(synthetic_data)
        
        logger.info(f"Constructed {len(grounding_data)} grounding data samples")
        
        return grounding_data
    
    async def _extract_grounding_from_step(self,
                                          step: Step,
                                          evaluation: Any,
                                          task: str) -> Optional[GroundingData]:
        """Extract grounding data from a single step."""
        
        if not step.post_state.get("screenshot"):
            return None
        
        # Extract UI elements
        ui_elements = self._extract_ui_elements(step.post_state)
        
        if not ui_elements:
            return None
        
        # Generate descriptions
        functional_desc = await self._generate_functional_descriptions(ui_elements, task)
        layout_desc = self._generate_layout_descriptions(ui_elements)
        fine_grained_text = self._extract_fine_grained_text(ui_elements)
        
        confidence = evaluation.confidence if hasattr(evaluation, 'confidence') else 0.5
        
        return GroundingData(
            screenshot=step.post_state["screenshot"],
            ui_elements=ui_elements,
            functional_descriptions=functional_desc,
            layout_descriptions=layout_desc,
            fine_grained_text=fine_grained_text,
            source="trajectory",
            confidence=confidence
        )
    
    def _extract_ui_elements(self, state: Dict) -> List[Dict[str, Any]]:
        """Extract UI elements from state."""
        
        elements = []
        
        # Extract from UI hierarchy if available
        if "ui_hierarchy" in state and isinstance(state["ui_hierarchy"], dict):
            hierarchy = state["ui_hierarchy"]
            
            # Parse elements (simplified - would be more complex in practice)
            if "elements" in hierarchy:
                for elem in hierarchy["elements"]:
                    elements.append({
                        "type": elem.get("type", "unknown"),
                        "text": elem.get("text", ""),
                        "bounds": elem.get("bounds", {}),
                        "clickable": elem.get("clickable", False),
                        "focused": elem.get("focused", False),
                        "description": elem.get("description", "")
                    })
        
        # Extract from windows
        if "windows" in state:
            for window in state["windows"]:
                elements.append({
                    "type": "window",
                    "text": window.get("title", ""),
                    "bounds": window.get("bounds", {}),
                    "clickable": False,
                    "focused": window.get("focused", False),
                    "description": f"Window: {window.get('title', 'Unknown')}"
                })
        
        return elements
    
    async def _generate_functional_descriptions(self,
                                               elements: List[Dict],
                                               task: str) -> List[str]:
        """Generate functional descriptions for UI elements."""
        
        descriptions = []
        
        for elem in elements:
            elem_type = elem.get("type", "")
            elem_text = elem.get("text", "")
            
            # Generate description based on element type and context
            if elem_type == "button":
                if elem.get("clickable"):
                    desc = f"Button '{elem_text}' - triggers action"
                else:
                    desc = f"Disabled button '{elem_text}'"
            elif elem_type == "input":
                desc = f"Input field for entering {elem.get('description', 'text')}"
            elif elem_type == "text":
                desc = f"Text label: '{elem_text[:50]}'"
            elif elem_type == "image":
                desc = f"Image element {elem.get('description', '')}"
            elif elem_type == "window":
                desc = f"Application window: {elem_text}"
            else:
                desc = f"{elem_type}: {elem_text[:30] if elem_text else 'no text'}"
            
            descriptions.append(desc)
        
        return descriptions
    
    def _generate_layout_descriptions(self, elements: List[Dict]) -> List[str]:
        """Generate spatial layout descriptions."""
        
        descriptions = []
        
        # Sort elements by position
        sorted_elements = sorted(
            elements,
            key=lambda e: (e.get("bounds", {}).get("top", 0), e.get("bounds", {}).get("left", 0))
        )
        
        # Generate relative position descriptions
        for i, elem in enumerate(sorted_elements):
            bounds = elem.get("bounds", {})
            
            if not bounds:
                continue
            
            desc_parts = []
            
            # Position on screen
            if bounds.get("top", 0) < 100:
                desc_parts.append("top of screen")
            elif bounds.get("bottom", 0) > 600:
                desc_parts.append("bottom of screen")
            else:
                desc_parts.append("middle of screen")
            
            if bounds.get("left", 0) < 100:
                desc_parts.append("left side")
            elif bounds.get("right", 0) > 300:
                desc_parts.append("right side")
            else:
                desc_parts.append("center")
            
            # Relative to other elements
            if i > 0:
                prev_elem = sorted_elements[i-1]
                prev_bounds = prev_elem.get("bounds", {})
                if prev_bounds.get("bottom", 0) < bounds.get("top", 0):
                    desc_parts.append(f"below {prev_elem.get('type', 'element')}")
            
            descriptions.append(f"{elem.get('type', 'Element')} at {', '.join(desc_parts)}")
        
        return descriptions[:10]  # Limit to top 10
    
    def _extract_fine_grained_text(self, elements: List[Dict]) -> List[Dict[str, Any]]:
        """Extract character/word-level text locations."""
        
        fine_grained = []
        
        for elem in elements:
            text = elem.get("text", "")
            if text and len(text) > 0:
                bounds = elem.get("bounds", {})
                
                # Word-level extraction (simplified)
                words = text.split()
                if words and bounds:
                    width = bounds.get("right", 0) - bounds.get("left", 0)
                    word_width = width / len(words) if words else 0
                    
                    for i, word in enumerate(words[:5]):  # Limit to 5 words
                        fine_grained.append({
                            "word": word,
                            "position": {
                                "left": bounds.get("left", 0) + i * word_width,
                                "top": bounds.get("top", 0),
                                "width": word_width,
                                "height": bounds.get("bottom", 0) - bounds.get("top", 0)
                            },
                            "element_type": elem.get("type", "text")
                        })
        
        return fine_grained
    
    async def _synthesize_grounding_data(self,
                                        trajectory_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]]) -> List[GroundingData]:
        """Synthesize additional grounding data using models."""
        
        synthetic_data = []
        
        # Select diverse screenshots for synthesis
        selected_screenshots = self._select_diverse_screenshots(trajectory_pairs, max_count=10)
        
        for screenshot, context in selected_screenshots:
            if self.model_client:
                # Use model to generate dense annotations
                prompt = self._build_grounding_synthesis_prompt(context)
                annotations = await self._call_model_for_grounding(prompt, screenshot)
                
                data = GroundingData(
                    screenshot=screenshot,
                    ui_elements=annotations.get("elements", []),
                    functional_descriptions=annotations.get("functions", []),
                    layout_descriptions=annotations.get("layout", []),
                    fine_grained_text=annotations.get("text_locations", []),
                    source="synthesis",
                    confidence=0.8
                )
                
                synthetic_data.append(data)
        
        return synthetic_data
    
    def _select_diverse_screenshots(self,
                                   trajectory_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]],
                                   max_count: int = 10) -> List[Tuple[Any, Dict]]:
        """Select diverse screenshots for synthesis."""
        
        selected = []
        seen_contexts = set()
        
        for trajectory, evaluation in trajectory_pairs:
            for step in trajectory.steps:
                if step.post_state.get("screenshot"):
                    # Create context signature
                    context_sig = f"{trajectory.query.platform}_{step.action.get('type')}"
                    
                    if context_sig not in seen_contexts:
                        selected.append((
                            step.post_state["screenshot"],
                            {
                                "task": trajectory.query.natural_instruction,
                                "platform": trajectory.query.platform,
                                "action": step.action
                            }
                        ))
                        seen_contexts.add(context_sig)
                        
                        if len(selected) >= max_count:
                            break
        
        return selected
    
    def _build_grounding_synthesis_prompt(self, context: Dict) -> str:
        """Build prompt for grounding data synthesis."""
        
        return f"""
        Analyze this UI screenshot and provide dense annotations:
        
        Context:
        - Task: {context.get('task', 'Unknown')}
        - Platform: {context.get('platform', 'Unknown')}
        - Recent action: {context.get('action', {})}
        
        Provide:
        1. All visible UI elements with types and bounds
        2. Functional description for each interactive element
        3. Spatial layout relationships
        4. Fine-grained text locations (word-level)
        """
    
    async def construct_planning_data(self,
                                     trajectory_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]]) -> List[TaskPlanningData]:
        """
        Construct task planning data from successful trajectories.
        """
        
        planning_data = []
        
        # Group trajectories by task type
        task_groups = defaultdict(list)
        for trajectory, evaluation in trajectory_pairs:
            if evaluation.is_correct:
                task_type = self._classify_task_type(trajectory.query.natural_instruction)
                task_groups[task_type].append((trajectory, evaluation))
        
        # Generate planning data for each group
        for task_type, group_pairs in task_groups.items():
            # Distill common patterns
            common_plan = await self._distill_execution_plan(group_pairs)
            
            if common_plan:
                planning_data.append(common_plan)
            
            # Generate variations
            for trajectory, evaluation in group_pairs[:5]:  # Limit to 5 per type
                individual_plan = await self._extract_individual_plan(
                    trajectory, evaluation
                )
                if individual_plan:
                    planning_data.append(individual_plan)
        
        # Generate LLM-based planning data if available
        if self.model_client:
            synthetic_plans = await self._generate_synthetic_plans(task_groups)
            planning_data.extend(synthetic_plans)
        
        logger.info(f"Constructed {len(planning_data)} planning data samples")
        
        return planning_data
    
    def _classify_task_type(self, task: str) -> str:
        """Classify task into categories."""
        
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["login", "sign in", "authenticate"]):
            return "authentication"
        elif any(word in task_lower for word in ["search", "find", "locate"]):
            return "search"
        elif any(word in task_lower for word in ["create", "add", "new"]):
            return "creation"
        elif any(word in task_lower for word in ["delete", "remove", "cancel"]):
            return "deletion"
        elif any(word in task_lower for word in ["navigate", "go to", "open"]):
            return "navigation"
        elif any(word in task_lower for word in ["edit", "modify", "update"]):
            return "modification"
        else:
            return "general"
    
    async def _distill_execution_plan(self,
                                     group_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]]) -> Optional[TaskPlanningData]:
        """Distill common execution pattern from multiple trajectories."""
        
        if not group_pairs:
            return None
        
        # Extract common steps
        all_steps = []
        for trajectory, evaluation in group_pairs:
            steps = []
            for i, (step, step_eval) in enumerate(zip(trajectory.steps, evaluation.step_evaluations)):
                if step_eval.annotation != CriticLabel.HARMFUL:
                    steps.append({
                        "action": step.action.get("type"),
                        "target": step.action.get("params", {}).get("element", ""),
                        "purpose": step_eval.summary,
                        "critical": i in evaluation.critical_steps
                    })
            all_steps.append(steps)
        
        # Find common pattern
        common_plan = self._find_common_pattern(all_steps)
        
        if not common_plan:
            return None
        
        # Extract key decisions
        key_decisions = self._extract_key_decisions(group_pairs)
        
        # Define conditions
        preconditions = self._extract_preconditions(group_pairs)
        postconditions = self._extract_postconditions(group_pairs)
        
        # Find alternatives
        alternatives = self._find_alternative_paths(all_steps, common_plan)
        
        return TaskPlanningData(
            task_description=group_pairs[0][0].query.natural_instruction,
            execution_plan=common_plan,
            key_decisions=key_decisions,
            preconditions=preconditions,
            postconditions=postconditions,
            alternative_paths=alternatives,
            source="trajectory_distillation",
            confidence=0.8
        )
    
    def _find_common_pattern(self, all_steps: List[List[Dict]]) -> List[Dict]:
        """Find common execution pattern across trajectories."""
        
        if not all_steps:
            return []
        
        # Simple approach: find most frequent action sequence
        # In practice, would use sequence alignment algorithms
        
        common = []
        min_length = min(len(steps) for steps in all_steps)
        
        for i in range(min_length):
            # Check if same action type at position i
            action_types = [steps[i]["action"] for steps in all_steps]
            
            # If majority agree
            most_common = max(set(action_types), key=action_types.count)
            if action_types.count(most_common) > len(action_types) / 2:
                # Find the step with this action
                for steps in all_steps:
                    if steps[i]["action"] == most_common:
                        common.append(steps[i])
                        break
        
        return common
    
    def _extract_key_decisions(self,
                              group_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]]) -> List[Dict]:
        """Extract key decision points from trajectories."""
        
        decisions = []
        
        for trajectory, evaluation in group_pairs:
            for idx in evaluation.critical_steps:
                if idx < len(trajectory.steps):
                    step = trajectory.steps[idx]
                    decisions.append({
                        "step_index": idx,
                        "decision": f"Execute {step.action.get('type')}",
                        "reason": f"Critical for task completion",
                        "alternatives": []  # Would extract from other trajectories
                    })
        
        # Deduplicate
        seen = set()
        unique_decisions = []
        for dec in decisions:
            key = f"{dec['decision']}_{dec['reason']}"
            if key not in seen:
                unique_decisions.append(dec)
                seen.add(key)
        
        return unique_decisions[:5]
    
    def _extract_preconditions(self,
                              group_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]]) -> List[str]:
        """Extract preconditions for task execution."""
        
        preconditions = []
        
        # Analyze initial states
        for trajectory, _ in group_pairs:
            if trajectory.steps:
                first_step = trajectory.steps[0]
                initial_state = first_step.pre_state
                
                # Extract conditions
                if initial_state.get("windows"):
                    window_titles = [w.get("title", "") for w in initial_state["windows"]]
                    if window_titles:
                        preconditions.append(f"Application '{window_titles[0]}' is open")
                
                if initial_state.get("ui_hierarchy", {}).get("element_count", 0) > 0:
                    preconditions.append("UI is loaded and responsive")
        
        # Common preconditions
        task = group_pairs[0][0].query.natural_instruction if group_pairs else ""
        if "login" in task.lower():
            preconditions.append("Login form is visible")
        elif "search" in task.lower():
            preconditions.append("Search interface is accessible")
        
        return list(set(preconditions))[:5]
    
    def _extract_postconditions(self,
                               group_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]]) -> List[str]:
        """Extract postconditions after task completion."""
        
        postconditions = []
        
        # Analyze final states
        for trajectory, evaluation in group_pairs:
            if trajectory.steps and evaluation.is_correct:
                last_step = trajectory.steps[-1]
                final_state = last_step.post_state
                
                # Extract conditions
                if final_state.get("windows"):
                    window_titles = [w.get("title", "") for w in final_state["windows"]]
                    if window_titles:
                        postconditions.append(f"Reached '{window_titles[0]}' screen")
        
        # Task-specific postconditions
        task = group_pairs[0][0].query.natural_instruction if group_pairs else ""
        if "login" in task.lower():
            postconditions.append("User is authenticated")
        elif "create" in task.lower():
            postconditions.append("New item is created and visible")
        elif "delete" in task.lower():
            postconditions.append("Target item is removed")
        
        return list(set(postconditions))[:5]
    
    def _find_alternative_paths(self,
                               all_steps: List[List[Dict]],
                               common_plan: List[Dict]) -> List[List[Dict]]:
        """Find alternative execution paths."""
        
        alternatives = []
        
        for steps in all_steps:
            # Check if significantly different from common plan
            if self._is_different_path(steps, common_plan):
                alternatives.append(steps[:10])  # Limit length
        
        return alternatives[:3]  # Max 3 alternatives
    
    def _is_different_path(self, steps: List[Dict], common: List[Dict]) -> bool:
        """Check if execution path is significantly different."""
        
        if abs(len(steps) - len(common)) > 3:
            return True
        
        differences = 0
        for i in range(min(len(steps), len(common))):
            if steps[i]["action"] != common[i]["action"]:
                differences += 1
        
        return differences > len(common) * 0.3
    
    async def _extract_individual_plan(self,
                                      trajectory: Trajectory,
                                      evaluation: EnhancedTrajectoryEvaluation) -> Optional[TaskPlanningData]:
        """Extract planning data from a single trajectory."""
        
        # Build execution plan
        execution_plan = []
        for i, (step, step_eval) in enumerate(zip(trajectory.steps, evaluation.step_evaluations)):
            if step_eval.annotation != CriticLabel.HARMFUL:
                execution_plan.append({
                    "step": i + 1,
                    "action": step.action.get("type"),
                    "target": step.action.get("params", {}).get("element", ""),
                    "description": step_eval.summary,
                    "result": "success" if step.success else "failure"
                })
        
        if not execution_plan:
            return None
        
        return TaskPlanningData(
            task_description=trajectory.query.natural_instruction,
            execution_plan=execution_plan,
            key_decisions=[],  # Simplified
            preconditions=[],
            postconditions=[],
            alternative_paths=[],
            source="trajectory",
            confidence=evaluation.confidence
        )
    
    async def _generate_synthetic_plans(self,
                                       task_groups: Dict[str, List]) -> List[TaskPlanningData]:
        """Generate synthetic planning data using LLM."""
        
        synthetic_plans = []
        
        for task_type, group_pairs in task_groups.items():
            if self.model_client and group_pairs:
                # Generate variations
                prompt = self._build_planning_synthesis_prompt(task_type, group_pairs[0][0])
                plan_text = await self._call_model_for_planning(prompt)
                
                # Parse plan
                plan = self._parse_synthetic_plan(plan_text, task_type)
                if plan:
                    synthetic_plans.append(plan)
        
        return synthetic_plans
    
    def _build_planning_synthesis_prompt(self, task_type: str, trajectory: Trajectory) -> str:
        """Build prompt for planning synthesis."""
        
        return f"""
        Generate a detailed execution plan for this type of task:
        
        Task Type: {task_type}
        Example: {trajectory.query.natural_instruction}
        Platform: {trajectory.query.platform}
        
        Provide:
        1. Step-by-step execution plan
        2. Key decision points
        3. Preconditions and postconditions
        4. Common failure points and recovery
        """
    
    def _parse_synthetic_plan(self, plan_text: str, task_type: str) -> Optional[TaskPlanningData]:
        """Parse LLM-generated plan text."""
        
        # Simple parsing - would be more sophisticated in practice
        lines = plan_text.split('\n')
        
        execution_plan = []
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                execution_plan.append({
                    "step": i + 1,
                    "description": line.strip()
                })
        
        if not execution_plan:
            return None
        
        return TaskPlanningData(
            task_description=f"Generic {task_type} task",
            execution_plan=execution_plan[:10],
            key_decisions=[],
            preconditions=[],
            postconditions=[],
            alternative_paths=[],
            source="llm_synthesis",
            confidence=0.7
        )
    
    async def construct_action_semantic_data(self,
                                            trajectory_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]]) -> List[ActionSemanticData]:
        """
        Construct action semantic understanding data.
        """
        
        semantic_data = []
        
        for trajectory, evaluation in trajectory_pairs:
            # Process consecutive steps
            for i in range(len(trajectory.steps) - 1):
                if (evaluation.step_evaluations[i].annotation != CriticLabel.HARMFUL and
                    trajectory.steps[i].post_state.get("screenshot") and
                    trajectory.steps[i+1].post_state.get("screenshot")):
                    
                    data = await self._extract_action_semantics(
                        trajectory.steps[i],
                        trajectory.steps[i+1],
                        evaluation.step_evaluations[i+1]
                    )
                    
                    if data:
                        semantic_data.append(data)
        
        logger.info(f"Constructed {len(semantic_data)} action semantic data samples")
        
        return semantic_data
    
    async def _extract_action_semantics(self,
                                       pre_step: Step,
                                       post_step: Step,
                                       post_evaluation: Any) -> Optional[ActionSemanticData]:
        """Extract semantic understanding of an action."""
        
        action = post_step.action
        
        # Generate descriptions
        if self.model_client:
            descriptions = await self._generate_action_descriptions(
                pre_step.post_state["screenshot"],
                post_step.post_state["screenshot"],
                action
            )
            action_desc = descriptions.get("description", "")
            expected = descriptions.get("expected", "")
            actual = descriptions.get("actual", "")
        else:
            action_desc = self._describe_action(action)
            expected = "State change expected"
            actual = "State changed" if post_step.success else "Action failed"
        
        # Categorize action
        category = self._categorize_action(action)
        
        confidence = post_evaluation.confidence if hasattr(post_evaluation, 'confidence') else 0.5
        
        return ActionSemanticData(
            pre_screenshot=pre_step.post_state["screenshot"],
            post_screenshot=post_step.post_state["screenshot"],
            action=action,
            action_description=action_desc,
            expected_effect=expected,
            actual_effect=actual,
            semantic_category=category,
            confidence=confidence
        )
    
    def _describe_action(self, action: Dict) -> str:
        """Generate natural language description of action."""
        
        action_type = action.get("type", "unknown")
        params = action.get("params", {})
        
        if action_type == "click":
            element = params.get("element", "element")
            return f"Click on {element}"
        elif action_type == "type":
            text = params.get("text", "")[:30]
            return f"Type '{text}'"
        elif action_type == "scroll":
            direction = params.get("direction", "down")
            return f"Scroll {direction}"
        else:
            return f"Perform {action_type} action"
    
    def _categorize_action(self, action: Dict) -> str:
        """Categorize action into semantic categories."""
        
        action_type = action.get("type", "").lower()
        
        if action_type in ["click", "tap", "press"]:
            return "selection"
        elif action_type in ["type", "input", "enter"]:
            return "input"
        elif action_type in ["scroll", "swipe"]:
            return "navigation"
        elif action_type in ["wait", "pause"]:
            return "timing"
        elif action_type in ["drag", "drop"]:
            return "manipulation"
        else:
            return "other"
    
    async def construct_reasoning_data(self,
                                      trajectory_pairs: List[Tuple[Trajectory, EnhancedTrajectoryEvaluation]]) -> List[ReasoningData]:
        """
        Construct chain-of-thought reasoning data.
        """
        
        reasoning_data = []
        
        # Different reasoning styles
        styles = ["concise", "detailed", "step-by-step", "goal-oriented"]
        
        for trajectory, evaluation in trajectory_pairs[:20]:  # Limit for performance
            if evaluation.is_correct:
                for style in styles:
                    data = await self._generate_reasoning(
                        trajectory, evaluation, style
                    )
                    if data:
                        reasoning_data.append(data)
        
        logger.info(f"Constructed {len(reasoning_data)} reasoning data samples")
        
        return reasoning_data
    
    async def _generate_reasoning(self,
                                 trajectory: Trajectory,
                                 evaluation: EnhancedTrajectoryEvaluation,
                                 style: str) -> Optional[ReasoningData]:
        """Generate reasoning data for a trajectory."""
        
        task = trajectory.query.natural_instruction
        
        # Generate reasoning steps
        reasoning_steps = []
        action_justifications = []
        state_interpretations = []
        goal_tracking = []
        
        for i, (step, step_eval) in enumerate(zip(trajectory.steps, evaluation.step_evaluations)):
            # Reasoning for this step
            reasoning = self._generate_step_reasoning(step, step_eval, task, style)
            if reasoning:
                reasoning_steps.append(reasoning)
            
            # Action justification
            justification = self._justify_action(step, task, i, len(trajectory.steps))
            action_justifications.append(justification)
            
            # State interpretation
            interpretation = self._interpret_state(step.post_state, task)
            state_interpretations.append(interpretation)
            
            # Goal tracking
            progress = self._track_goal_progress(i, len(trajectory.steps), step_eval)
            goal_tracking.append(progress)
        
        if not reasoning_steps:
            return None
        
        return ReasoningData(
            task=task,
            reasoning_steps=reasoning_steps[:10],  # Limit length
            action_justifications=action_justifications[:10],
            state_interpretations=state_interpretations[:10],
            goal_tracking=goal_tracking[:10],
            style=style,
            source="trajectory_analysis"
        )
    
    def _generate_step_reasoning(self,
                                step: Step,
                                evaluation: Any,
                                task: str,
                                style: str) -> str:
        """Generate reasoning for a single step."""
        
        action_type = step.action.get("type", "")
        
        if style == "concise":
            return f"Need to {action_type} to proceed"
        elif style == "detailed":
            return f"At this point, I need to {action_type} because {evaluation.summary if hasattr(evaluation, 'summary') else 'it advances the task'}"
        elif style == "step-by-step":
            return f"Step {step.index + 1}: {action_type} - {evaluation.summary if hasattr(evaluation, 'summary') else 'continue task'}"
        elif style == "goal-oriented":
            return f"To achieve '{task}', now {action_type}"
        else:
            return f"Execute {action_type}"
    
    def _justify_action(self, step: Step, task: str, index: int, total: int) -> str:
        """Justify why an action was taken."""
        
        action_type = step.action.get("type", "")
        progress = (index + 1) / total * 100
        
        if index == 0:
            return f"Starting with {action_type} to begin the task"
        elif index == total - 1:
            return f"Final {action_type} to complete the task"
        elif progress < 33:
            return f"Early {action_type} to set up for main task"
        elif progress < 66:
            return f"Core {action_type} for main task execution"
        else:
            return f"Finalizing with {action_type}"
    
    def _interpret_state(self, state: Dict, task: str) -> str:
        """Interpret current UI state."""
        
        if state.get("windows"):
            window = state["windows"][0] if state["windows"] else {}
            return f"Currently on '{window.get('title', 'unknown')}' screen"
        elif state.get("ui_hierarchy", {}).get("element_count", 0) > 0:
            count = state["ui_hierarchy"]["element_count"]
            return f"UI loaded with {count} elements available"
        else:
            return "Waiting for UI to update"
    
    def _track_goal_progress(self, index: int, total: int, evaluation: Any) -> str:
        """Track progress toward goal."""
        
        progress = (index + 1) / total * 100
        
        if hasattr(evaluation, 'annotation'):
            if evaluation.annotation == CriticLabel.GOOD:
                return f"Progress: {progress:.0f}% - on track"
            elif evaluation.annotation == CriticLabel.HARMFUL:
                return f"Progress: {progress:.0f}% - encountered issue"
            else:
                return f"Progress: {progress:.0f}% - continuing"
        else:
            return f"Progress: {progress:.0f}%"
    
    async def _generate_action_descriptions(self,
                                           pre_screenshot: Any,
                                           post_screenshot: Any,
                                           action: Dict) -> Dict[str, str]:
        """Generate action descriptions using VLM."""
        
        # Placeholder for VLM call
        return {
            "description": self._describe_action(action),
            "expected": "UI should respond to action",
            "actual": "UI updated as expected"
        }
    
    async def _call_model_for_grounding(self, prompt: str, screenshot: Any) -> Dict:
        """Call model for grounding annotations."""
        
        # Placeholder
        return {
            "elements": [],
            "functions": [],
            "layout": [],
            "text_locations": []
        }
    
    async def _call_model_for_planning(self, prompt: str) -> str:
        """Call model for planning generation."""
        
        # Placeholder
        return "1. Open application\n2. Navigate to target\n3. Perform action\n4. Verify result"
    
    def export_data(self, output_dir: str):
        """Export all constructed data to files."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export each data type
        for data_type, data_list in self.data_cache.items():
            filepath = os.path.join(output_dir, f"{data_type}_data.json")
            
            # Convert to serializable format
            serializable_data = []
            for item in data_list:
                if hasattr(item, '__dict__'):
                    # Convert dataclass to dict
                    item_dict = {
                        k: v for k, v in item.__dict__.items()
                        if not k.startswith('_') and v is not None
                    }
                    # Remove screenshot data for JSON serialization
                    if 'screenshot' in item_dict:
                        item_dict['screenshot'] = "binary_data"
                    if 'pre_screenshot' in item_dict:
                        item_dict['pre_screenshot'] = "binary_data"
                    if 'post_screenshot' in item_dict:
                        item_dict['post_screenshot'] = "binary_data"
                    
                    serializable_data.append(item_dict)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Exported {len(serializable_data)} {data_type} samples to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about constructed data."""
        
        stats = {}
        
        for data_type, data_list in self.data_cache.items():
            stats[data_type] = {
                "count": len(data_list),
                "sources": defaultdict(int)
            }
            
            # Count by source
            for item in data_list:
                if hasattr(item, 'source'):
                    stats[data_type]["sources"][item.source] += 1
        
        return stats