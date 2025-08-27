"""
Action Executor - 统一的动作执行器
"""
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import asyncio

from agentbay.session import Session
from .config import Platform, ActionType

logger = logging.getLogger(__name__)


class ActionExecutor:
    """统一的动作执行器，处理不同平台的动作执行"""
    
    def __init__(self, session: Session, platform: Platform):
        self.session = session
        self.platform = platform
        self.last_action_result = None
        
    async def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行动作并返回结果"""
        action_type = action.get("type", "").lower()
        params = action.get("params", {})
        
        try:
            if self.platform == Platform.MOBILE:
                return await self._execute_mobile_action(action_type, params)
            elif self.platform == Platform.PC:
                return await self._execute_pc_action(action_type, params)
            elif self.platform == Platform.WEB:
                return await self._execute_web_action(action_type, params)
            else:
                return {"success": False, "error": f"Unsupported platform: {self.platform}"}
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            return {"success": False, "error": str(e)}
            
    async def _execute_mobile_action(self, action_type: str, params: Dict) -> Dict[str, Any]:
        """执行Android移动端动作"""
        
        command_map = {
            "click": self._android_click,
            "tap": self._android_click,
            "type": self._android_type,
            "text": self._android_type,
            "scroll": self._android_scroll,
            "swipe": self._android_swipe,
            "long_press": self._android_long_press,
            "back": self._android_back,
            "home": self._android_home,
            "recent": self._android_recent
        }
        
        handler = command_map.get(action_type)
        if handler:
            return await handler(params)
        else:
            return {"success": False, "error": f"Unknown mobile action: {action_type}"}
            
    async def _android_click(self, params: Dict) -> Dict[str, Any]:
        """Android点击操作"""
        x = params.get("x", 0)
        y = params.get("y", 0)
        
        result = await self.session.command.execute(f"input tap {x} {y}")
        return self._format_result(result)
        
    async def _android_type(self, params: Dict) -> Dict[str, Any]:
        """Android输入文本"""
        text = params.get("text", "")
        # 转义特殊字符
        text = text.replace("'", "\\'").replace('"', '\\"').replace(" ", "%s")
        
        result = await self.session.command.execute(f"input text '{text}'")
        return self._format_result(result)
        
    async def _android_scroll(self, params: Dict) -> Dict[str, Any]:
        """Android滚动操作"""
        x1 = params.get("x1", params.get("x", 500))
        y1 = params.get("y1", params.get("start_y", 1000))
        x2 = params.get("x2", x1)
        y2 = params.get("y2", params.get("end_y", 500))
        duration = params.get("duration", 300)
        
        result = await self.session.command.execute(f"input swipe {x1} {y1} {x2} {y2} {duration}")
        return self._format_result(result)
        
    async def _android_swipe(self, params: Dict) -> Dict[str, Any]:
        """Android滑动操作"""
        return await self._android_scroll(params)
        
    async def _android_long_press(self, params: Dict) -> Dict[str, Any]:
        """Android长按操作"""
        x = params.get("x", 0)
        y = params.get("y", 0)
        duration = params.get("duration", 1000)
        
        result = await self.session.command.execute(f"input swipe {x} {y} {x} {y} {duration}")
        return self._format_result(result)
        
    async def _android_back(self, params: Dict) -> Dict[str, Any]:
        """Android返回键"""
        result = await self.session.command.execute("input keyevent KEYCODE_BACK")
        return self._format_result(result)
        
    async def _android_home(self, params: Dict) -> Dict[str, Any]:
        """Android主页键"""
        result = await self.session.command.execute("input keyevent KEYCODE_HOME")
        return self._format_result(result)
        
    async def _android_recent(self, params: Dict) -> Dict[str, Any]:
        """Android最近任务键"""
        result = await self.session.command.execute("input keyevent KEYCODE_APP_SWITCH")
        return self._format_result(result)
        
    async def _execute_pc_action(self, action_type: str, params: Dict) -> Dict[str, Any]:
        """执行PC桌面端动作"""
        
        command_map = {
            "click": self._pc_click,
            "double_click": self._pc_double_click,
            "right_click": self._pc_right_click,
            "type": self._pc_type,
            "key": self._pc_key,
            "key_press": self._pc_key,
            "drag": self._pc_drag,
            "move": self._pc_move,
            "scroll": self._pc_scroll
        }
        
        handler = command_map.get(action_type)
        if handler:
            return await handler(params)
        else:
            return {"success": False, "error": f"Unknown PC action: {action_type}"}
            
    async def _pc_click(self, params: Dict) -> Dict[str, Any]:
        """PC鼠标点击"""
        x = params.get("x", 0)
        y = params.get("y", 0)
        button = params.get("button", 1)
        
        result = await self.session.command.execute(f"xdotool mousemove {x} {y} click {button}")
        return self._format_result(result)
        
    async def _pc_double_click(self, params: Dict) -> Dict[str, Any]:
        """PC双击"""
        x = params.get("x", 0)
        y = params.get("y", 0)
        
        result = await self.session.command.execute(f"xdotool mousemove {x} {y} click --repeat 2 1")
        return self._format_result(result)
        
    async def _pc_right_click(self, params: Dict) -> Dict[str, Any]:
        """PC右键点击"""
        x = params.get("x", 0)
        y = params.get("y", 0)
        
        result = await self.session.command.execute(f"xdotool mousemove {x} {y} click 3")
        return self._format_result(result)
        
    async def _pc_type(self, params: Dict) -> Dict[str, Any]:
        """PC键盘输入"""
        text = params.get("text", "")
        
        result = await self.session.command.execute(f"xdotool type '{text}'")
        return self._format_result(result)
        
    async def _pc_key(self, params: Dict) -> Dict[str, Any]:
        """PC按键"""
        key = params.get("key", "")
        modifiers = params.get("modifiers", [])
        
        if modifiers:
            modifier_str = "+".join(modifiers)
            result = await self.session.command.execute(f"xdotool key {modifier_str}+{key}")
        else:
            result = await self.session.command.execute(f"xdotool key {key}")
            
        return self._format_result(result)
        
    async def _pc_drag(self, params: Dict) -> Dict[str, Any]:
        """PC拖拽操作"""
        x1 = params.get("x1", params.get("from_x", 0))
        y1 = params.get("y1", params.get("from_y", 0))
        x2 = params.get("x2", params.get("to_x", 0))
        y2 = params.get("y2", params.get("to_y", 0))
        
        commands = [
            f"xdotool mousemove {x1} {y1}",
            "xdotool mousedown 1",
            f"xdotool mousemove {x2} {y2}",
            "xdotool mouseup 1"
        ]
        
        for cmd in commands:
            result = await self.session.command.execute(cmd)
            if not result.success:
                return self._format_result(result)
                
        return {"success": True, "output": "Drag completed"}
        
    async def _pc_move(self, params: Dict) -> Dict[str, Any]:
        """PC鼠标移动"""
        x = params.get("x", 0)
        y = params.get("y", 0)
        
        result = await self.session.command.execute(f"xdotool mousemove {x} {y}")
        return self._format_result(result)
        
    async def _pc_scroll(self, params: Dict) -> Dict[str, Any]:
        """PC滚动操作"""
        direction = params.get("direction", "down")
        amount = params.get("amount", 3)
        
        if direction == "up":
            button = 4
        else:
            button = 5
            
        result = await self.session.command.execute(f"xdotool click --repeat {amount} {button}")
        return self._format_result(result)
        
    async def _execute_web_action(self, action_type: str, params: Dict) -> Dict[str, Any]:
        """执行Web浏览器动作"""
        
        if not hasattr(self.session, 'browser'):
            return {"success": False, "error": "Browser not available in session"}
            
        command_map = {
            "click": self._web_click,
            "type": self._web_type,
            "navigate": self._web_navigate,
            "goto": self._web_navigate,
            "back": self._web_back,
            "forward": self._web_forward,
            "refresh": self._web_refresh,
            "scroll": self._web_scroll,
            "select": self._web_select,
            "wait": self._web_wait,
            "evaluate": self._web_evaluate
        }
        
        handler = command_map.get(action_type)
        if handler:
            return await handler(params)
        else:
            return {"success": False, "error": f"Unknown web action: {action_type}"}
            
    async def _web_click(self, params: Dict) -> Dict[str, Any]:
        """Web点击操作"""
        selector = params.get("selector")
        x = params.get("x")
        y = params.get("y")
        
        try:
            if selector:
                result = await self.session.browser.click(selector)
            elif x is not None and y is not None:
                result = await self.session.browser.click_at(x, y)
            else:
                return {"success": False, "error": "Either selector or x,y coordinates required"}
                
            return self._format_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _web_type(self, params: Dict) -> Dict[str, Any]:
        """Web输入文本"""
        selector = params.get("selector", "")
        text = params.get("text", "")
        clear = params.get("clear", False)
        
        try:
            if clear and selector:
                # 先清空输入框
                await self.session.browser.clear(selector)
                
            result = await self.session.browser.type(selector, text)
            return self._format_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _web_navigate(self, params: Dict) -> Dict[str, Any]:
        """Web导航到URL"""
        url = params.get("url", "")
        wait_until = params.get("wait_until", "load")
        
        try:
            result = await self.session.browser.goto(url, wait_until=wait_until)
            return self._format_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _web_back(self, params: Dict) -> Dict[str, Any]:
        """Web后退"""
        try:
            result = await self.session.browser.go_back()
            return self._format_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _web_forward(self, params: Dict) -> Dict[str, Any]:
        """Web前进"""
        try:
            result = await self.session.browser.go_forward()
            return self._format_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _web_refresh(self, params: Dict) -> Dict[str, Any]:
        """Web刷新页面"""
        try:
            result = await self.session.browser.reload()
            return self._format_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _web_scroll(self, params: Dict) -> Dict[str, Any]:
        """Web滚动操作"""
        direction = params.get("direction", "down")
        amount = params.get("amount", 300)
        
        try:
            if direction == "down":
                script = f"window.scrollBy(0, {amount})"
            elif direction == "up":
                script = f"window.scrollBy(0, -{amount})"
            elif direction == "right":
                script = f"window.scrollBy({amount}, 0)"
            elif direction == "left":
                script = f"window.scrollBy(-{amount}, 0)"
            else:
                script = f"window.scrollTo({params.get('x', 0)}, {params.get('y', 0)})"
                
            result = await self.session.browser.evaluate(script)
            return self._format_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _web_select(self, params: Dict) -> Dict[str, Any]:
        """Web选择下拉框选项"""
        selector = params.get("selector", "")
        value = params.get("value")
        index = params.get("index")
        text = params.get("text")
        
        try:
            if value is not None:
                result = await self.session.browser.select_option(selector, value=value)
            elif index is not None:
                result = await self.session.browser.select_option(selector, index=index)
            elif text is not None:
                result = await self.session.browser.select_option(selector, label=text)
            else:
                return {"success": False, "error": "Either value, index, or text required"}
                
            return self._format_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _web_wait(self, params: Dict) -> Dict[str, Any]:
        """Web等待元素或条件"""
        selector = params.get("selector")
        timeout = params.get("timeout", 5000)
        state = params.get("state", "visible")
        
        try:
            if selector:
                result = await self.session.browser.wait_for_selector(
                    selector, 
                    timeout=timeout,
                    state=state
                )
                return self._format_result(result)
            else:
                # 简单的延时等待
                await asyncio.sleep(timeout / 1000)
                return {"success": True, "output": f"Waited {timeout}ms"}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _web_evaluate(self, params: Dict) -> Dict[str, Any]:
        """Web执行JavaScript"""
        script = params.get("script", "")
        
        try:
            result = await self.session.browser.evaluate(script)
            return self._format_result(result)
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _format_result(self, result) -> Dict[str, Any]:
        """格式化执行结果"""
        if hasattr(result, 'success'):
            return {
                "success": result.success,
                "output": getattr(result.data, "output", "") if result.success else "",
                "error": result.error if not result.success else None
            }
        else:
            return {"success": True, "output": str(result)}