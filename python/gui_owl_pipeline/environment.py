import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional
import asyncio
import logging
from dataclasses import dataclass
from agentbay.agent_bay import AgentBay
from .config import PipelineConfig, Platform, OperatingSystem

logger = logging.getLogger(__name__)

@dataclass
class VirtualEnvironment:
    session_id: str
    platform: Platform
    os: OperatingSystem
    session: Any
    is_active: bool = True
    metadata: Dict[str, Any] = None

class EnvironmentInfrastructure:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = AgentBay(api_key=config.api_key)
        self.environments: Dict[str, VirtualEnvironment] = {}
        self.available_pool: List[str] = []
        
    async def initialize_pool(self, size: int = None):
        size = size or self.config.parallel_sessions
        logger.info(f"Initializing environment pool with {size} sessions")
        
        tasks = []
        for i in range(size):
            tasks.append(self._create_environment(f"env_{i}"))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to create environment {i}: {result}")
            else:
                self.available_pool.append(result.session_id)
                
        logger.info(f"Successfully initialized {len(self.available_pool)} environments")
        
    async def _create_environment(self, env_id: str) -> VirtualEnvironment:
        session_result = await self.client.create_session(
            platform=self._get_platform_config()
        )
        
        if not session_result.success:
            raise Exception(f"Failed to create session: {session_result.error}")
            
        env = VirtualEnvironment(
            session_id=session_result.data.session_id,
            platform=self.config.platform,
            os=self.config.operating_system,
            session=session_result.data,
            metadata={
                "created_at": asyncio.get_event_loop().time(),
                "env_id": env_id
            }
        )
        
        self.environments[session_result.data.session_id] = env
        logger.info(f"Created environment {env_id} with session {session_result.data.session_id}")
        
        return env
        
    def _get_platform_config(self) -> Dict[str, Any]:
        configs = {
            Platform.MOBILE: {
                "platform": "android",
                "version": "13",
                "device": "pixel_6"
            },
            Platform.PC: {
                "platform": self.config.operating_system.value,
                "resolution": "1920x1080"
            },
            Platform.WEB: {
                "platform": "browser",
                "browser": "chrome",
                "headless": False
            }
        }
        return configs.get(self.config.platform, {})
        
    async def acquire_environment(self) -> Optional[VirtualEnvironment]:
        if not self.available_pool:
            logger.warning("No available environments in pool")
            try:
                env = await self._create_environment(f"env_dynamic_{len(self.environments)}")
                return env
            except Exception as e:
                logger.error(f"Failed to create dynamic environment: {e}")
                return None
                
        session_id = self.available_pool.pop(0)
        env = self.environments.get(session_id)
        
        if env and env.is_active:
            logger.info(f"Acquired environment {session_id}")
            return env
        else:
            logger.warning(f"Environment {session_id} is not active")
            return await self.acquire_environment()
            
    async def release_environment(self, session_id: str):
        if session_id in self.environments:
            env = self.environments[session_id]
            if env.is_active:
                self.available_pool.append(session_id)
                logger.info(f"Released environment {session_id} back to pool")
                
    async def reset_environment(self, session_id: str):
        env = self.environments.get(session_id)
        if not env:
            logger.error(f"Environment {session_id} not found")
            return False
            
        try:
            if self.config.platform == Platform.MOBILE:
                await self._reset_mobile(env)
            elif self.config.platform == Platform.PC:
                await self._reset_pc(env)
            else:
                await self._reset_web(env)
                
            logger.info(f"Successfully reset environment {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset environment {session_id}: {e}")
            return False
            
    async def _reset_mobile(self, env: VirtualEnvironment):
        session = self.client.get_session(env.session_id)
        await session.command.execute("input keyevent KEYCODE_HOME")
        await session.command.execute("pm clear-data")
        
    async def _reset_pc(self, env: VirtualEnvironment):
        session = self.client.get_session(env.session_id)
        if env.os == OperatingSystem.WINDOWS:
            await session.command.execute("taskkill /F /IM *")
        else:
            await session.command.execute("pkill -9 -f .")
            
    async def _reset_web(self, env: VirtualEnvironment):
        session = self.client.get_session(env.session_id)
        if hasattr(session, 'browser'):
            await session.browser.clear_cookies()
            await session.browser.goto("about:blank")
            
    async def capture_state(self, session_id: str) -> Dict[str, Any]:
        env = self.environments.get(session_id)
        if not env:
            logger.error(f"Environment {session_id} not found")
            return {}
            
        session = self.client.get_session(session_id)
        state = {}
        
        try:
            screenshot_result = await session.filesystem.screenshot()
            if screenshot_result.success:
                state["screenshot"] = screenshot_result.data
                
            if self.config.platform == Platform.MOBILE:
                ui_result = await session.command.execute("uiautomator dump /dev/stdout")
                if ui_result.success:
                    state["ui_hierarchy"] = ui_result.data.output
                    
            elif self.config.platform == Platform.PC:
                if hasattr(session, 'application'):
                    windows = await session.application.list_windows()
                    if windows.success:
                        state["windows"] = windows.data
                        
            state["timestamp"] = asyncio.get_event_loop().time()
            state["session_id"] = session_id
            
        except Exception as e:
            logger.error(f"Failed to capture state for {session_id}: {e}")
            
        return state
        
    async def execute_action(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        env = self.environments.get(session_id)
        if not env:
            logger.error(f"Environment {session_id} not found")
            return {"success": False, "error": "Environment not found"}
            
        session = self.client.get_session(session_id)
        result = {"success": False}
        
        try:
            action_type = action.get("type")
            params = action.get("params", {})
            
            if action_type == "click":
                x, y = params.get("x"), params.get("y")
                if self.config.platform == Platform.MOBILE:
                    cmd_result = await session.command.execute(f"input tap {x} {y}")
                else:
                    if hasattr(session, 'browser'):
                        await session.browser.click(x, y)
                        cmd_result = {"success": True}
                    else:
                        cmd_result = await session.command.execute(f"xdotool mousemove {x} {y} click 1")
                        
            elif action_type == "type":
                text = params.get("text", "")
                if self.config.platform == Platform.MOBILE:
                    cmd_result = await session.command.execute(f"input text '{text}'")
                else:
                    if hasattr(session, 'browser'):
                        await session.browser.type(text)
                        cmd_result = {"success": True}
                    else:
                        cmd_result = await session.command.execute(f"xdotool type '{text}'")
                        
            elif action_type == "scroll":
                direction = params.get("direction", "down")
                amount = params.get("amount", 500)
                if self.config.platform == Platform.MOBILE:
                    if direction == "down":
                        cmd_result = await session.command.execute(f"input swipe 500 1000 500 {1000-amount}")
                    else:
                        cmd_result = await session.command.execute(f"input swipe 500 500 500 {500+amount}")
                else:
                    if hasattr(session, 'browser'):
                        await session.browser.scroll(0, amount if direction == "down" else -amount)
                        cmd_result = {"success": True}
                        
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return {"success": False, "error": f"Unknown action type: {action_type}"}
                
            if hasattr(cmd_result, 'success'):
                result["success"] = cmd_result.success
                if cmd_result.success and hasattr(cmd_result, 'data'):
                    result["output"] = cmd_result.data
            else:
                result["success"] = True
                
        except Exception as e:
            logger.error(f"Failed to execute action on {session_id}: {e}")
            result["error"] = str(e)
            
        return result
        
    async def cleanup(self):
        logger.info("Cleaning up all environments")
        
        tasks = []
        for session_id in list(self.environments.keys()):
            tasks.append(self._cleanup_environment(session_id))
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.environments.clear()
        self.available_pool.clear()
        
    async def _cleanup_environment(self, session_id: str):
        try:
            result = await self.client.delete_session(session_id)
            if result.success:
                logger.info(f"Successfully deleted session {session_id}")
            else:
                logger.error(f"Failed to delete session {session_id}: {result.error}")
        except Exception as e:
            logger.error(f"Exception while deleting session {session_id}: {e}")
            
    def get_environment_stats(self) -> Dict[str, Any]:
        active_envs = sum(1 for env in self.environments.values() if env.is_active)
        
        return {
            "total_environments": len(self.environments),
            "active_environments": active_envs,
            "available_pool_size": len(self.available_pool),
            "platform": self.config.platform.value,
            "operating_system": self.config.operating_system.value
        }