"""
增强版轨迹收集器 - 深度集成Wuying AgentBay SDK
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import base64
from datetime import datetime
import time

from agentbay.agent_bay import AgentBay
from agentbay.session import Session
from agentbay.command import Command
from agentbay.filesystem import FileSystem
from agentbay.browser import Browser
from agentbay.application import Application

from .config import PipelineConfig, Platform, ActionType
from .query_generator import Query
from .action_executor import ActionExecutor
from .ui_parser import UIParser

logger = logging.getLogger(__name__)

@dataclass
class AgentBayStep:
    index: int
    action: Dict[str, Any]
    pre_screenshot: bytes
    post_screenshot: bytes
    ui_data: Optional[Dict] = None
    window_info: Optional[Dict] = None
    browser_state: Optional[Dict] = None
    execution_result: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

@dataclass
class AgentBayTrajectory:
    id: str
    query: Query
    session_id: str
    steps: List[AgentBayStep]
    session_info: Dict[str, Any]
    platform_info: Dict[str, Any]
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    
class AgentBayCollector:
    def __init__(self, config: PipelineConfig):
        self.config = config
        if not config.api_key:
            raise ValueError("API key is required for AgentBay collector")
        self.client = AgentBay(api_key=config.api_key)
        self.active_sessions: Dict[str, Session] = {}
        self.session_lock = asyncio.Lock()
        
    async def create_session_for_platform(self, platform: Platform) -> Session:
        """根据平台创建合适的会话"""
        max_retries = self.config.max_retries
        for attempt in range(max_retries):
            try:
                if platform == Platform.MOBILE:
                    # Android环境
                    result = await self.client.create_session(
                        image="android:13",
                        resources={"cpu": 2, "memory": "4Gi"},
                        timeout=self.config.step_timeout * self.config.max_trajectory_length
                    )
                elif platform == Platform.PC:
                    # Windows/Linux桌面环境
                    result = await self.client.create_session(
                        image="ubuntu:22.04-desktop",
                        resources={"cpu": 4, "memory": "8Gi"},
                        timeout=self.config.step_timeout * self.config.max_trajectory_length,
                        enable_vnc=True
                    )
                else:  # Platform.WEB
                    # 浏览器环境
                    result = await self.client.create_session(
                        image="browser:chrome-latest",
                        resources={"cpu": 2, "memory": "4Gi"},
                        timeout=self.config.step_timeout * self.config.max_trajectory_length,
                        enable_browser=True
                    )
                    
                if result.success:
                    session = self.client.get_session(result.data.session_id)
                    async with self.session_lock:
                        self.active_sessions[result.data.session_id] = session
                    logger.info(f"Created session {result.data.session_id} for {platform.value}")
                    return session
                else:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to create session: {result.error}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed with exception: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        raise Exception(f"Failed to create session after {max_retries} attempts")
            
    async def collect_trajectory_with_agentbay(self, query: Query) -> AgentBayTrajectory:
        """使用AgentBay SDK收集完整轨迹"""
        session = None
        trajectory = None
        
        try:
            # 创建会话
            session = await self.create_session_for_platform(query.platform)
            
            trajectory = AgentBayTrajectory(
                id=f"traj_{query.id}_{datetime.now().timestamp()}",
                query=query,
                session_id=session.session_id,
                steps=[],
                session_info={"session_id": session.session_id},
                platform_info={"platform": query.platform.value}
            )
            
            # 初始化环境
            await self._initialize_environment(session, query.platform)
            
            # 收集轨迹
            for i in range(self.config.max_trajectory_length):
                try:
                    step = await asyncio.wait_for(
                        self._collect_single_step(session, query, i, trajectory.steps),
                        timeout=self.config.step_timeout
                    )
                    
                    if step:
                        trajectory.steps.append(step)
                        
                        # 检查是否完成
                        if await self._check_task_completion(session, query):
                            logger.info(f"Task completed at step {i}")
                            trajectory.success = True
                            break
                    else:
                        logger.warning(f"Failed to collect step {i}")
                        if i == 0:
                            trajectory.error_message = "Failed to collect initial step"
                            break
                except asyncio.TimeoutError:
                    logger.error(f"Step {i} timed out after {self.config.step_timeout} seconds")
                    trajectory.error_message = f"Timeout at step {i}"
                    break
                except Exception as e:
                    logger.error(f"Error collecting step {i}: {e}")
                    trajectory.error_message = f"Error at step {i}: {str(e)}"
                    break
                    
        except Exception as e:
            logger.error(f"Error during trajectory collection: {e}")
            if trajectory:
                trajectory.error_message = str(e)
            else:
                raise
        finally:
            if trajectory:
                trajectory.end_time = time.time()
            # 清理会话
            if session:
                await self._cleanup_session(session.session_id)
            
        return trajectory
        
    async def _initialize_environment(self, session: Session, platform: Platform):
        """初始化环境"""
        
        if platform == Platform.MOBILE:
            # Android初始化
            await session.command.execute("am start -a android.intent.action.MAIN -c android.intent.category.HOME")
            await asyncio.sleep(2)
            
        elif platform == Platform.PC:
            # 桌面初始化
            await session.command.execute("export DISPLAY=:0")
            await session.command.execute("startx &")
            await asyncio.sleep(5)
            
        else:  # WEB
            # 浏览器初始化
            if hasattr(session, 'browser'):
                await session.browser.goto("about:blank")
                await asyncio.sleep(1)
                
    async def _collect_single_step(self, session: Session, query: Query, step_index: int, history: List[AgentBayStep]) -> Optional[AgentBayStep]:
        """收集单个步骤的数据"""
        
        # 截图前状态
        pre_screenshot = await self._capture_screenshot(session)
        
        # 获取UI数据
        ui_data = await self._get_ui_data(session, query.platform)
        
        # 预测下一个动作
        action = await self._predict_action(query, ui_data, history)
        
        if not action:
            return None
            
        # 执行动作
        execution_result = await self._execute_action(session, action, query.platform)
        
        # 等待UI更新
        await asyncio.sleep(self.config.screenshot_interval)
        
        # 截图后状态
        post_screenshot = await self._capture_screenshot(session)
        
        # 获取额外信息
        window_info = None
        browser_state = None
        
        if query.platform == Platform.PC:
            window_info = await self._get_window_info(session)
        elif query.platform == Platform.WEB:
            browser_state = await self._get_browser_state(session)
            
        step = AgentBayStep(
            index=step_index,
            action=action,
            pre_screenshot=pre_screenshot,
            post_screenshot=post_screenshot,
            ui_data=ui_data,
            window_info=window_info,
            browser_state=browser_state,
            execution_result=execution_result,
            timestamp=asyncio.get_event_loop().time()
        )
        
        return step
        
    async def _capture_screenshot(self, session: Session) -> bytes:
        """捕获截图"""
        try:
            result = await session.filesystem.screenshot()
            
            if result.success:
                # 如果是base64编码，解码
                if isinstance(result.data, str):
                    return base64.b64decode(result.data)
                elif isinstance(result.data, bytes):
                    return result.data
                else:
                    logger.warning(f"Unexpected screenshot data type: {type(result.data)}")
                    return b""
            else:
                logger.error(f"Failed to capture screenshot: {result.error}")
                return b""
        except Exception as e:
            logger.error(f"Exception capturing screenshot: {e}")
            return b""
            
    async def _get_ui_data(self, session: Session, platform: Platform) -> Optional[Dict]:
        """获取UI层次结构数据"""
        ui_parser = UIParser(session, platform)
        return await ui_parser.get_ui_hierarchy()
            
    async def _predict_action(self, query: Query, ui_data: Optional[Dict], history: List[AgentBayStep]) -> Optional[Dict]:
        """预测下一个动作"""
        
        # 这里应该调用实际的模型
        # 现在使用简单的规则
        
        if not ui_data:
            return None
            
        if query.platform == Platform.MOBILE:
            elements = ui_data.get("elements", [])
            
            # 查找可点击元素
            for elem in elements:
                if elem.get("clickable"):
                    bounds = elem.get("bounds", "")
                    if bounds:
                        # 解析bounds: [x1,y1][x2,y2]
                        import re
                        coords = re.findall(r'\d+', bounds)
                        if len(coords) >= 4:
                            x = (int(coords[0]) + int(coords[2])) // 2
                            y = (int(coords[1]) + int(coords[3])) // 2
                            
                            return {
                                "type": "click",
                                "params": {"x": x, "y": y, "element": elem.get("text", "")}
                            }
                            
        # 默认动作
        return {
            "type": "click",
            "params": {"x": 500, "y": 500}
        }
        
    async def _execute_action(self, session: Session, action: Dict, platform: Platform) -> Dict:
        """执行动作"""
        executor = ActionExecutor(session, platform)
        return await executor.execute(action)
            
    async def _get_window_info(self, session: Session) -> Optional[Dict]:
        """获取窗口信息"""
        if hasattr(session, 'application'):
            result = await session.application.list_windows()
            if result.success:
                return {"windows": result.data}
        return None
        
    async def _get_browser_state(self, session: Session) -> Optional[Dict]:
        """获取浏览器状态"""
        if hasattr(session, 'browser'):
            url_result = await session.browser.get_url()
            title_result = await session.browser.get_title()
            
            state = {}
            if url_result.success:
                state["url"] = url_result.data
            if title_result.success:
                state["title"] = title_result.data
                
            return state
        return None
        
    async def _check_task_completion(self, session: Session, query: Query) -> bool:
        """检查任务是否完成"""
        
        # 这里应该根据query的目标来判断
        # 现在使用简单的规则
        
        if query.platform == Platform.WEB:
            if hasattr(session, 'browser'):
                # 检查是否到达目标页面
                url_result = await session.browser.get_url()
                if url_result.success:
                    # 简单检查URL是否包含某些关键词
                    return any(keyword in url_result.data for keyword in ["success", "complete", "done"])
                    
        # 默认在达到一定步数后完成
        return False
        
    async def _cleanup_session(self, session_id: str):
        """清理会话"""
        try:
            async with self.session_lock:
                if session_id in self.active_sessions:
                    result = await self.client.delete_session(session_id)
                    if result.success:
                        del self.active_sessions[session_id]
                        logger.info(f"Cleaned up session {session_id}")
                    else:
                        logger.error(f"Failed to clean up session {session_id}: {result.error}")
        except Exception as e:
            logger.error(f"Exception during session cleanup: {e}")
                
    async def collect_batch_with_agentbay(self, queries: List[Query]) -> List[AgentBayTrajectory]:
        """批量收集轨迹"""
        
        # 限制并发数量
        semaphore = asyncio.Semaphore(self.config.parallel_sessions)
        
        async def collect_with_limit(query: Query):
            async with semaphore:
                return await self.collect_trajectory_with_agentbay(query)
        
        tasks = []
        for query in queries:
            task = asyncio.create_task(collect_with_limit(query))
            tasks.append(task)
            
        trajectories = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_trajectories = []
        for i, traj in enumerate(trajectories):
            if isinstance(traj, Exception):
                logger.error(f"Failed to collect trajectory for query {queries[i].id}: {traj}")
            elif traj:
                valid_trajectories.append(traj)
                logger.info(f"Successfully collected trajectory for query {queries[i].id}, success={traj.success}")
                
        return valid_trajectories
        
    def export_agentbay_trajectory(self, trajectory: AgentBayTrajectory, output_dir: str):
        """导出AgentBay轨迹数据"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存轨迹元数据
        metadata = {
            "id": trajectory.id,
            "query": {
                "id": trajectory.query.id,
                "instruction": trajectory.query.natural_instruction,
                "platform": trajectory.query.platform.value
            },
            "session_id": trajectory.session_id,
            "session_info": trajectory.session_info,
            "platform_info": trajectory.platform_info,
            "steps": []
        }
        
        # 保存每个步骤
        screenshots_dir = os.path.join(output_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        for step in trajectory.steps:
            # 保存截图
            pre_path = os.path.join(screenshots_dir, f"step_{step.index}_pre.png")
            post_path = os.path.join(screenshots_dir, f"step_{step.index}_post.png")
            
            with open(pre_path, 'wb') as f:
                f.write(step.pre_screenshot)
            with open(post_path, 'wb') as f:
                f.write(step.post_screenshot)
                
            # 添加步骤信息
            metadata["steps"].append({
                "index": step.index,
                "action": step.action,
                "pre_screenshot": pre_path,
                "post_screenshot": post_path,
                "ui_data": step.ui_data,
                "window_info": step.window_info,
                "browser_state": step.browser_state,
                "execution_result": step.execution_result,
                "timestamp": step.timestamp
            })
            
        # 保存元数据
        with open(os.path.join(output_dir, "trajectory.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Exported AgentBay trajectory to {output_dir}")