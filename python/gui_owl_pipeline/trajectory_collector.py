import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from .config import ActionType, PipelineConfig
from .environment import EnvironmentInfrastructure
from .query_generator import Query

logger = logging.getLogger(__name__)

@dataclass
class Step:
    index: int
    action: Dict[str, Any]
    pre_state: Dict[str, Any]
    post_state: Dict[str, Any]
    timestamp: float
    execution_time: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trajectory:
    id: str
    query: Query
    session_id: str
    steps: List[Step]
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        return self.status in ["completed", "failed", "timeout"]
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
        
    @property
    def success_rate(self) -> float:
        if not self.steps:
            return 0
        successful = sum(1 for step in self.steps if step.success)
        return successful / len(self.steps)

class TrajectoryCollector:
    def __init__(self, config: PipelineConfig, env_infrastructure: EnvironmentInfrastructure, model_client=None):
        self.config = config
        self.env_infrastructure = env_infrastructure
        self.model_client = model_client
        self.trajectories: Dict[str, Trajectory] = {}
        self.active_collections: Dict[str, asyncio.Task] = {}
        
    async def collect_trajectory(self, query: Query, guidance: Optional[Dict] = None) -> Trajectory:
        env = await self.env_infrastructure.acquire_environment()
        if not env:
            logger.error(f"Failed to acquire environment for query {query.id}")
            return None
            
        trajectory = Trajectory(
            id=f"traj_{query.id}_{datetime.now().timestamp()}",
            query=query,
            session_id=env.session_id,
            steps=[],
            start_time=datetime.now(),
            end_time=None,
            status="in_progress",
            metadata={
                "guidance": guidance,
                "platform": query.platform.value
            }
        )
        
        self.trajectories[trajectory.id] = trajectory
        
        try:
            await self._collect_steps(trajectory, env.session_id, guidance)
            trajectory.status = "completed"
            
        except asyncio.TimeoutError:
            logger.warning(f"Trajectory {trajectory.id} timed out")
            trajectory.status = "timeout"
            
        except Exception as e:
            logger.error(f"Error collecting trajectory {trajectory.id}: {e}")
            trajectory.status = "failed"
            trajectory.metadata["error"] = str(e)
            
        finally:
            trajectory.end_time = datetime.now()
            await self.env_infrastructure.release_environment(env.session_id)
            
        return trajectory
        
    async def _collect_steps(self, trajectory: Trajectory, session_id: str, guidance: Optional[Dict]):
        current_state = await self.env_infrastructure.capture_state(session_id)
        
        for i in range(self.config.max_trajectory_length):
            action = await self._predict_next_action(
                trajectory.query,
                current_state,
                trajectory.steps,
                guidance
            )
            
            if action is None or action.get("type") == "done":
                logger.info(f"Trajectory {trajectory.id} completed at step {i}")
                break
                
            pre_state = current_state.copy()
            
            start_time = asyncio.get_event_loop().time()
            result = await self.env_infrastructure.execute_action(session_id, action)
            execution_time = asyncio.get_event_loop().time() - start_time
            
            await asyncio.sleep(self.config.screenshot_interval)
            
            post_state = await self.env_infrastructure.capture_state(session_id)
            
            step = Step(
                index=i,
                action=action,
                pre_state=pre_state,
                post_state=post_state,
                timestamp=asyncio.get_event_loop().time(),
                execution_time=execution_time,
                success=result.get("success", False),
                error=result.get("error"),
                metadata={
                    "output": result.get("output"),
                    "guidance_used": guidance is not None
                }
            )
            
            trajectory.steps.append(step)
            current_state = post_state
            
            if not step.success:
                logger.warning(f"Step {i} failed in trajectory {trajectory.id}: {step.error}")
                if self._should_retry(step):
                    await self._handle_retry(trajectory, session_id, step)
                else:
                    break
                    
    async def _predict_next_action(self, query: Query, state: Dict, history: List[Step], guidance: Optional[Dict]) -> Optional[Dict]:
        if self.model_client:
            context = self._build_model_context(query, state, history, guidance)
            
            prediction = await self._call_model(context)
            
            return self._parse_model_prediction(prediction)
            
        else:
            return self._rule_based_prediction(query, state, history)
            
    def _build_model_context(self, query: Query, state: Dict, history: List[Step], guidance: Optional[Dict]) -> Dict:
        context = {
            "instruction": query.natural_instruction,
            "current_state": {
                "screenshot": state.get("screenshot"),
                "ui_hierarchy": state.get("ui_hierarchy"),
                "windows": state.get("windows")
            },
            "history": []
        }
        
        for step in history[-5:]:
            context["history"].append({
                "action": step.action,
                "success": step.success,
                "error": step.error
            })
            
        if guidance:
            context["guidance"] = guidance.get("steps", [])
            
        return context
        
    async def _call_model(self, context: Dict) -> Dict:
        return {
            "action": {
                "type": "click",
                "params": {"x": 100, "y": 200}
            },
            "confidence": 0.85,
            "reasoning": "Clicking on the target element"
        }
        
    def _parse_model_prediction(self, prediction: Dict) -> Optional[Dict]:
        if prediction.get("confidence", 0) < 0.5:
            return None
            
        action = prediction.get("action", {})
        
        if not action.get("type"):
            return None
            
        return action
        
    def _rule_based_prediction(self, query: Query, state: Dict, history: List[Step]) -> Optional[Dict]:
        if len(history) >= len(query.path):
            return {"type": "done"}
            
        step_index = len(history)
        
        if step_index < len(query.path):
            page_name = query.path[step_index]
            
            return {
                "type": "click",
                "params": {
                    "x": 500,
                    "y": 300 + (step_index * 100),
                    "element": page_name
                }
            }
            
        return {"type": "done"}
        
    def _should_retry(self, step: Step) -> bool:
        if step.error and "timeout" in step.error.lower():
            return True
            
        if step.error and "network" in step.error.lower():
            return True
            
        return False
        
    async def _handle_retry(self, trajectory: Trajectory, session_id: str, failed_step: Step):
        logger.info(f"Retrying failed step {failed_step.index} in trajectory {trajectory.id}")
        
        await asyncio.sleep(2)
        
        result = await self.env_infrastructure.execute_action(session_id, failed_step.action)
        
        if result.get("success"):
            failed_step.success = True
            failed_step.error = None
            failed_step.metadata["retried"] = True
            logger.info(f"Retry successful for step {failed_step.index}")
        else:
            logger.warning(f"Retry failed for step {failed_step.index}")
            
    async def collect_batch_trajectories(self, queries: List[Query], guidance_map: Optional[Dict] = None) -> List[Trajectory]:
        tasks = []
        
        for query in queries:
            guidance = guidance_map.get(query.id) if guidance_map else None
            task = asyncio.create_task(self.collect_trajectory(query, guidance))
            tasks.append(task)
            self.active_collections[query.id] = task
            
        trajectories = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_trajectories = []
        for i, traj in enumerate(trajectories):
            if isinstance(traj, Exception):
                logger.error(f"Failed to collect trajectory for query {queries[i].id}: {traj}")
            elif traj:
                valid_trajectories.append(traj)
                
        for query in queries:
            if query.id in self.active_collections:
                del self.active_collections[query.id]
                
        return valid_trajectories
        
    def export_trajectory(self, trajectory: Trajectory, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        traj_data = {
            "id": trajectory.id,
            "query": {
                "id": trajectory.query.id,
                "instruction": trajectory.query.natural_instruction,
                "path": trajectory.query.path,
                "difficulty": trajectory.query.difficulty
            },
            "session_id": trajectory.session_id,
            "duration": trajectory.duration,
            "success_rate": trajectory.success_rate,
            "status": trajectory.status,
            "steps": []
        }
        
        for step in trajectory.steps:
            step_data = {
                "index": step.index,
                "action": step.action,
                "success": step.success,
                "error": step.error,
                "execution_time": step.execution_time
            }
            
            if step.pre_state.get("screenshot"):
                screenshot_path = os.path.join(output_dir, f"step_{step.index}_pre.png")
                self._save_screenshot(step.pre_state["screenshot"], screenshot_path)
                step_data["pre_screenshot"] = screenshot_path
                
            if step.post_state.get("screenshot"):
                screenshot_path = os.path.join(output_dir, f"step_{step.index}_post.png")
                self._save_screenshot(step.post_state["screenshot"], screenshot_path)
                step_data["post_screenshot"] = screenshot_path
                
            traj_data["steps"].append(step_data)
            
        output_file = os.path.join(output_dir, f"{trajectory.id}.json")
        with open(output_file, 'w') as f:
            json.dump(traj_data, f, indent=2)
            
        logger.info(f"Exported trajectory to {output_file}")
        
    def _save_screenshot(self, screenshot_data: Any, path: str):
        try:
            if isinstance(screenshot_data, bytes):
                with open(path, 'wb') as f:
                    f.write(screenshot_data)
            elif isinstance(screenshot_data, str):
                import base64
                data = base64.b64decode(screenshot_data)
                with open(path, 'wb') as f:
                    f.write(data)
        except Exception as e:
            logger.error(f"Failed to save screenshot to {path}: {e}")
            
    def get_statistics(self) -> Dict[str, Any]:
        total = len(self.trajectories)
        completed = sum(1 for t in self.trajectories.values() if t.status == "completed")
        failed = sum(1 for t in self.trajectories.values() if t.status == "failed")
        timeout = sum(1 for t in self.trajectories.values() if t.status == "timeout")
        in_progress = sum(1 for t in self.trajectories.values() if t.status == "in_progress")
        
        avg_duration = 0
        avg_steps = 0
        avg_success_rate = 0
        
        if completed > 0:
            completed_trajs = [t for t in self.trajectories.values() if t.status == "completed"]
            avg_duration = sum(t.duration for t in completed_trajs) / len(completed_trajs)
            avg_steps = sum(len(t.steps) for t in completed_trajs) / len(completed_trajs)
            avg_success_rate = sum(t.success_rate for t in completed_trajs) / len(completed_trajs)
            
        return {
            "total_trajectories": total,
            "completed": completed,
            "failed": failed,
            "timeout": timeout,
            "in_progress": in_progress,
            "active_collections": len(self.active_collections),
            "avg_duration": avg_duration,
            "avg_steps": avg_steps,
            "avg_success_rate": avg_success_rate
        }