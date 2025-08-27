import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime
from .config import PipelineConfig, Platform
from .environment import EnvironmentInfrastructure
from .query_generator import QueryGenerator, Query, Page, PageTransition
from .trajectory_collector import TrajectoryCollector, Trajectory
from .trajectory_evaluator import TrajectoryLevelCritic, TrajectoryEvaluation
from .guidance_generator import GuidanceGenerator, QueryGuidance

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    total_queries: int
    total_trajectories: int
    successful_trajectories: int
    failed_trajectories: int
    high_quality_trajectories: int
    guidance_generated: int
    duration: float
    metadata: Dict[str, Any] = None

class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.env_infrastructure = EnvironmentInfrastructure(config)
        self.query_generator = QueryGenerator()
        self.trajectory_collector = TrajectoryCollector(config, self.env_infrastructure)
        self.trajectory_evaluator = TrajectoryLevelCritic(config)
        self.guidance_generator = GuidanceGenerator(config)
        
        self.queries: List[Query] = []
        self.trajectories: Dict[str, Trajectory] = {}
        self.evaluations: Dict[str, TrajectoryEvaluation] = {}
        self.guidance: Dict[str, QueryGuidance] = {}
        
        self.iteration_count = 0
        self.max_iterations = 5
        
    async def initialize(self):
        logger.info("Initializing data pipeline")
        await self.env_infrastructure.initialize_pool(self.config.parallel_sessions)
        
    async def run_mobile_pipeline(self, app_name: str, pages: List[Page], transitions: List[PageTransition], query_count: int = 10) -> PipelineResult:
        
        start_time = datetime.now()
        
        dag = self.query_generator.build_dag_graph(pages, transitions)
        
        self.queries = self.query_generator.generate_mobile_queries(app_name, dag, query_count)
        logger.info(f"Generated {len(self.queries)} mobile queries")
        
        result = await self._run_self_evolving_loop()
        
        duration = (datetime.now() - start_time).total_seconds()
        result.duration = duration
        
        return result
        
    async def run_pc_pipeline(self, software_name: str, skills: List[Dict], query_count: int = 10) -> PipelineResult:
        
        start_time = datetime.now()
        
        self.queries = self.query_generator.generate_pc_queries(software_name, skills, query_count)
        logger.info(f"Generated {len(self.queries)} PC queries")
        
        result = await self._run_self_evolving_loop()
        
        duration = (datetime.now() - start_time).total_seconds()
        result.duration = duration
        
        return result
        
    async def _run_self_evolving_loop(self) -> PipelineResult:
        
        all_trajectories = []
        all_evaluations = []
        
        for iteration in range(self.max_iterations):
            logger.info(f"Starting self-evolution iteration {iteration + 1}/{self.max_iterations}")
            
            trajectories = await self.trajectory_collector.collect_batch_trajectories(
                self.queries, 
                self.guidance
            )
            
            evaluations = await self.trajectory_evaluator.evaluate_batch(trajectories)
            
            high_quality = self.trajectory_evaluator.filter_by_quality(evaluations)
            
            for traj, eval in zip(trajectories, evaluations):
                self.trajectories[traj.id] = traj
                self.evaluations[traj.id] = eval
                
            all_trajectories.extend(trajectories)
            all_evaluations.extend(evaluations)
            
            difficult_queries = self._identify_difficult_queries(evaluations)
            
            new_guidance = await self.guidance_generator.generate_batch_guidance(
                difficult_queries,
                {q.id: self._get_failed_trajectory(q.id) for q in difficult_queries},
                {q.id: self._get_evaluation(q.id) for q in difficult_queries}
            )
            
            self.guidance.update(new_guidance)
            
            logger.info(f"Iteration {iteration + 1} complete: {len(high_quality)} high-quality trajectories")
            
            if self._should_stop_evolution(high_quality, all_evaluations):
                logger.info("Stopping self-evolution: quality threshold reached")
                break
                
        return self._compile_results(all_trajectories, all_evaluations)
        
    def _identify_difficult_queries(self, evaluations: List[TrajectoryEvaluation]) -> List[Query]:
        difficult = []
        
        for eval in evaluations:
            if not eval.is_correct or eval.confidence < 0.6:
                query_id = eval.trajectory_id.split('_')[1]
                query = next((q for q in self.queries if q.id == query_id), None)
                if query and query not in difficult:
                    difficult.append(query)
                    
        return difficult
        
    def _get_failed_trajectory(self, query_id: str) -> Optional[Trajectory]:
        for traj_id, traj in self.trajectories.items():
            if query_id in traj_id and traj.status != "completed":
                return traj
        return None
        
    def _get_evaluation(self, query_id: str) -> Optional[TrajectoryEvaluation]:
        for eval_id, eval in self.evaluations.items():
            if query_id in eval_id:
                return eval
        return None
        
    def _should_stop_evolution(self, high_quality: List[TrajectoryEvaluation], all_evaluations: List[TrajectoryEvaluation]) -> bool:
        
        if not all_evaluations:
            return False
            
        quality_ratio = len(high_quality) / len(all_evaluations)
        
        if quality_ratio > 0.8:
            return True
            
        if self.iteration_count >= self.max_iterations:
            return True
            
        return False
        
    def _compile_results(self, trajectories: List[Trajectory], evaluations: List[TrajectoryEvaluation]) -> PipelineResult:
        
        successful = sum(1 for t in trajectories if t.status == "completed")
        failed = sum(1 for t in trajectories if t.status == "failed")
        high_quality = sum(1 for e in evaluations if e.is_correct and e.confidence > self.config.critic_threshold)
        
        result = PipelineResult(
            total_queries=len(self.queries),
            total_trajectories=len(trajectories),
            successful_trajectories=successful,
            failed_trajectories=failed,
            high_quality_trajectories=high_quality,
            guidance_generated=len(self.guidance),
            duration=0,
            metadata={
                "iterations": self.iteration_count,
                "platform": self.config.platform.value,
                "avg_trajectory_length": sum(len(t.steps) for t in trajectories) / len(trajectories) if trajectories else 0,
                "avg_success_rate": sum(t.success_rate for t in trajectories) / len(trajectories) if trajectories else 0
            }
        )
        
        return result
        
    async def generate_downstream_data(self):
        
        logger.info("Generating downstream data from collected trajectories")
        
        grounding_data = await self._generate_grounding_data()
        
        planning_data = await self._generate_planning_data()
        
        action_semantic_data = await self._generate_action_semantic_data()
        
        return {
            "grounding": grounding_data,
            "planning": planning_data,
            "action_semantics": action_semantic_data
        }
        
    async def _generate_grounding_data(self) -> List[Dict]:
        grounding_data = []
        
        for traj in self.trajectories.values():
            for step in traj.steps:
                if step.post_state.get("ui_hierarchy"):
                    grounding_data.append({
                        "screenshot": step.post_state.get("screenshot"),
                        "ui_elements": self._extract_ui_elements(step.post_state),
                        "action": step.action,
                        "timestamp": step.timestamp
                    })
                    
        logger.info(f"Generated {len(grounding_data)} grounding data samples")
        return grounding_data
        
    async def _generate_planning_data(self) -> List[Dict]:
        planning_data = []
        
        successful_trajectories = [t for t in self.trajectories.values() if t.status == "completed"]
        
        for traj in successful_trajectories:
            plan = {
                "task": traj.query.natural_instruction,
                "steps": [],
                "key_decisions": []
            }
            
            for step in traj.steps:
                plan["steps"].append({
                    "action": step.action.get("type"),
                    "target": step.action.get("params", {}).get("element", "unknown"),
                    "result": "success" if step.success else "failure"
                })
                
            planning_data.append(plan)
            
        logger.info(f"Generated {len(planning_data)} planning data samples")
        return planning_data
        
    async def _generate_action_semantic_data(self) -> List[Dict]:
        semantic_data = []
        
        for traj in self.trajectories.values():
            for i, step in enumerate(traj.steps):
                if i > 0:
                    semantic_data.append({
                        "pre_screenshot": traj.steps[i-1].post_state.get("screenshot"),
                        "post_screenshot": step.post_state.get("screenshot"),
                        "action": step.action,
                        "description": f"Execute {step.action.get('type')} to progress task"
                    })
                    
        logger.info(f"Generated {len(semantic_data)} action semantic data samples")
        return semantic_data
        
    def _extract_ui_elements(self, state: Dict) -> List[Dict]:
        elements = []
        
        return elements
        
    def export_pipeline_data(self, output_dir: str):
        
        os.makedirs(output_dir, exist_ok=True)
        
        queries_dir = os.path.join(output_dir, "queries")
        os.makedirs(queries_dir, exist_ok=True)
        self.query_generator.export_queries(self.queries, os.path.join(queries_dir, "queries.json"))
        
        trajectories_dir = os.path.join(output_dir, "trajectories")
        os.makedirs(trajectories_dir, exist_ok=True)
        for traj in self.trajectories.values():
            traj_dir = os.path.join(trajectories_dir, traj.id)
            self.trajectory_collector.export_trajectory(traj, traj_dir)
            
        evaluations_file = os.path.join(output_dir, "evaluations.json")
        self._export_evaluations(evaluations_file)
        
        guidance_dir = os.path.join(output_dir, "guidance")
        os.makedirs(guidance_dir, exist_ok=True)
        for query_id, guide in self.guidance.items():
            self.guidance_generator.export_guidance(
                guide, 
                os.path.join(guidance_dir, f"{query_id}_guidance.json")
            )
            
        logger.info(f"Exported all pipeline data to {output_dir}")
        
    def _export_evaluations(self, output_file: str):
        data = []
        
        for eval_id, eval in self.evaluations.items():
            data.append({
                "trajectory_id": eval.trajectory_id,
                "is_correct": eval.is_correct,
                "confidence": eval.confidence,
                "consensus_result": eval.consensus_result,
                "text_channel": eval.text_channel_result,
                "multimodal_channel": eval.multimodal_channel_result,
                "metadata": eval.metadata
            })
            
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    async def cleanup(self):
        logger.info("Cleaning up data pipeline resources")
        await self.env_infrastructure.cleanup()
        
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "pipeline": {
                "total_queries": len(self.queries),
                "total_trajectories": len(self.trajectories),
                "total_evaluations": len(self.evaluations),
                "guidance_generated": len(self.guidance),
                "iterations_completed": self.iteration_count
            },
            "environment": self.env_infrastructure.get_environment_stats(),
            "collector": self.trajectory_collector.get_statistics(),
            "quality": {
                "high_quality_count": sum(1 for e in self.evaluations.values() if e.is_correct and e.confidence > self.config.critic_threshold),
                "avg_confidence": sum(e.confidence for e in self.evaluations.values()) / len(self.evaluations) if self.evaluations else 0
            }
        }