"""
UI Parser - 统一的UI数据解析器
"""
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
import json
import re

from agentbay.session import Session
from .config import Platform

logger = logging.getLogger(__name__)


class UIElement:
    """UI元素的统一表示"""
    
    def __init__(self, element_type: str, text: str = "", bounds: Tuple[int, int, int, int] = None,
                 clickable: bool = False, scrollable: bool = False, enabled: bool = True,
                 attributes: Dict = None):
        self.element_type = element_type
        self.text = text
        self.bounds = bounds or (0, 0, 0, 0)
        self.clickable = clickable
        self.scrollable = scrollable
        self.enabled = enabled
        self.attributes = attributes or {}
        
    @property
    def center(self) -> Tuple[int, int]:
        """获取元素中心点坐标"""
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) // 2, (y1 + y2) // 2)
        
    @property
    def width(self) -> int:
        """获取元素宽度"""
        return self.bounds[2] - self.bounds[0]
        
    @property
    def height(self) -> int:
        """获取元素高度"""
        return self.bounds[3] - self.bounds[1]
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "type": self.element_type,
            "text": self.text,
            "bounds": self.bounds,
            "center": self.center,
            "clickable": self.clickable,
            "scrollable": self.scrollable,
            "enabled": self.enabled,
            "attributes": self.attributes
        }


class UIParser:
    """统一的UI数据解析器"""
    
    def __init__(self, session: Session, platform: Platform):
        self.session = session
        self.platform = platform
        
    async def get_ui_hierarchy(self) -> Optional[Dict[str, Any]]:
        """获取UI层次结构"""
        
        try:
            if self.platform == Platform.MOBILE:
                return await self._get_android_ui()
            elif self.platform == Platform.PC:
                return await self._get_desktop_ui()
            elif self.platform == Platform.WEB:
                return await self._get_web_ui()
            else:
                logger.error(f"Unsupported platform: {self.platform}")
                return None
        except Exception as e:
            logger.error(f"Error getting UI hierarchy: {e}")
            return None
            
    async def _get_android_ui(self) -> Optional[Dict]:
        """获取Android UI数据"""
        
        try:
            # 执行UI dump
            result = await self.session.command.execute("uiautomator dump /sdcard/ui.xml")
            if not result.success:
                logger.error(f"Failed to dump Android UI: {result.error}")
                return None
                
            # 读取XML内容
            result = await self.session.command.execute("cat /sdcard/ui.xml")
            if not result.success:
                logger.error(f"Failed to read Android UI XML: {result.error}")
                return None
                
            xml_content = result.data.output
            elements = self._parse_android_xml(xml_content)
            
            # 获取屏幕信息
            screen_info = await self._get_android_screen_info()
            
            return {
                "platform": "android",
                "elements": [elem.to_dict() for elem in elements],
                "screen": screen_info,
                "raw_xml": xml_content
            }
            
        except Exception as e:
            logger.error(f"Error getting Android UI: {e}")
            return None
            
    def _parse_android_xml(self, xml_content: str) -> List[UIElement]:
        """解析Android UI XML"""
        
        elements = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for node in root.iter():
                # 提取属性
                element_type = node.get('class', '').split('.')[-1]
                text = node.get('text', '')
                resource_id = node.get('resource-id', '')
                content_desc = node.get('content-desc', '')
                clickable = node.get('clickable') == 'true'
                scrollable = node.get('scrollable') == 'true'
                enabled = node.get('enabled') == 'true'
                bounds_str = node.get('bounds', '')
                
                # 解析bounds
                bounds = self._parse_bounds(bounds_str)
                
                # 只添加有意义的元素
                if bounds and (clickable or scrollable or text or content_desc):
                    elem = UIElement(
                        element_type=element_type,
                        text=text or content_desc,
                        bounds=bounds,
                        clickable=clickable,
                        scrollable=scrollable,
                        enabled=enabled,
                        attributes={
                            'resource_id': resource_id,
                            'content_desc': content_desc,
                            'package': node.get('package', ''),
                            'checkable': node.get('checkable') == 'true',
                            'checked': node.get('checked') == 'true',
                            'focusable': node.get('focusable') == 'true',
                            'focused': node.get('focused') == 'true',
                            'selected': node.get('selected') == 'true'
                        }
                    )
                    elements.append(elem)
                    
        except Exception as e:
            logger.error(f"Error parsing Android XML: {e}")
            
        return elements
        
    def _parse_bounds(self, bounds_str: str) -> Optional[Tuple[int, int, int, int]]:
        """解析bounds字符串 [x1,y1][x2,y2]"""
        
        if not bounds_str:
            return None
            
        try:
            coords = re.findall(r'\d+', bounds_str)
            if len(coords) >= 4:
                return (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
        except Exception as e:
            logger.error(f"Error parsing bounds: {e}")
            
        return None
        
    async def _get_android_screen_info(self) -> Dict:
        """获取Android屏幕信息"""
        
        try:
            result = await self.session.command.execute("wm size")
            if result.success:
                match = re.search(r'(\d+)x(\d+)', result.data.output)
                if match:
                    width, height = int(match.group(1)), int(match.group(2))
                    return {"width": width, "height": height}
        except Exception as e:
            logger.error(f"Error getting Android screen info: {e}")
            
        return {"width": 1080, "height": 1920}  # 默认值
        
    async def _get_desktop_ui(self) -> Optional[Dict]:
        """获取桌面UI数据"""
        
        try:
            # 使用accessibility API
            if hasattr(self.session, 'application'):
                result = await self.session.application.get_accessibility_tree()
                if result.success:
                    elements = self._parse_desktop_accessibility(result.data)
                    
                    # 获取窗口信息
                    window_info = await self._get_window_info()
                    
                    return {
                        "platform": "desktop",
                        "elements": [elem.to_dict() for elem in elements],
                        "windows": window_info,
                        "raw_data": result.data
                    }
                    
            # 备用方案：使用AT-SPI2 (Linux)
            result = await self.session.command.execute("atspi2-info")
            if result.success:
                return self._parse_atspi_output(result.data.output)
                
        except Exception as e:
            logger.error(f"Error getting desktop UI: {e}")
            
        return None
        
    def _parse_desktop_accessibility(self, data: Any) -> List[UIElement]:
        """解析桌面accessibility数据"""
        
        elements = []
        
        # 根据实际的accessibility API返回格式解析
        # 这里需要根据具体的API响应格式来实现
        
        return elements
        
    async def _get_window_info(self) -> List[Dict]:
        """获取窗口信息"""
        
        windows = []
        
        try:
            if hasattr(self.session, 'application'):
                result = await self.session.application.list_windows()
                if result.success:
                    return result.data
                    
            # 备用方案：使用wmctrl
            result = await self.session.command.execute("wmctrl -l -G")
            if result.success:
                for line in result.data.output.split('\n'):
                    parts = line.split()
                    if len(parts) >= 8:
                        windows.append({
                            "id": parts[0],
                            "desktop": parts[1],
                            "x": int(parts[2]),
                            "y": int(parts[3]),
                            "width": int(parts[4]),
                            "height": int(parts[5]),
                            "title": ' '.join(parts[7:])
                        })
                        
        except Exception as e:
            logger.error(f"Error getting window info: {e}")
            
        return windows
        
    async def _get_web_ui(self) -> Optional[Dict]:
        """获取Web UI数据"""
        
        try:
            if not hasattr(self.session, 'browser'):
                logger.error("Browser not available in session")
                return None
                
            # 获取DOM结构
            dom_result = await self.session.browser.evaluate("""
                function getElements() {
                    const elements = [];
                    const allElements = document.querySelectorAll('*');
                    
                    allElements.forEach(elem => {
                        const rect = elem.getBoundingClientRect();
                        const isVisible = rect.width > 0 && rect.height > 0;
                        const isInteractive = elem.tagName === 'A' || elem.tagName === 'BUTTON' || 
                                            elem.tagName === 'INPUT' || elem.tagName === 'SELECT' ||
                                            elem.tagName === 'TEXTAREA' || elem.onclick || 
                                            elem.getAttribute('role') === 'button';
                        
                        if (isVisible && (isInteractive || elem.textContent.trim())) {
                            elements.push({
                                tagName: elem.tagName,
                                text: elem.textContent.trim().substring(0, 100),
                                bounds: [rect.left, rect.top, rect.right, rect.bottom],
                                clickable: isInteractive,
                                attributes: {
                                    id: elem.id,
                                    className: elem.className,
                                    href: elem.href,
                                    role: elem.getAttribute('role'),
                                    ariaLabel: elem.getAttribute('aria-label')
                                }
                            });
                        }
                    });
                    
                    return elements;
                }
                
                getElements();
            """)
            
            if not dom_result.success:
                logger.error(f"Failed to get DOM elements: {dom_result.error}")
                return None
                
            # 获取页面信息
            page_info = await self._get_page_info()
            
            # 转换为UIElement对象
            elements = []
            for elem_data in dom_result.data:
                elem = UIElement(
                    element_type=elem_data.get('tagName', ''),
                    text=elem_data.get('text', ''),
                    bounds=tuple(elem_data.get('bounds', [0, 0, 0, 0])),
                    clickable=elem_data.get('clickable', False),
                    attributes=elem_data.get('attributes', {})
                )
                elements.append(elem)
                
            return {
                "platform": "web",
                "elements": [elem.to_dict() for elem in elements],
                "page": page_info,
                "element_count": len(elements)
            }
            
        except Exception as e:
            logger.error(f"Error getting web UI: {e}")
            return None
            
    async def _get_page_info(self) -> Dict:
        """获取页面信息"""
        
        info = {}
        
        try:
            if hasattr(self.session, 'browser'):
                # 获取URL
                url_result = await self.session.browser.get_url()
                if url_result.success:
                    info["url"] = url_result.data
                    
                # 获取标题
                title_result = await self.session.browser.get_title()
                if title_result.success:
                    info["title"] = title_result.data
                    
                # 获取视口大小
                viewport_result = await self.session.browser.evaluate("""
                    ({
                        width: window.innerWidth,
                        height: window.innerHeight,
                        scrollX: window.scrollX,
                        scrollY: window.scrollY,
                        scrollWidth: document.body.scrollWidth,
                        scrollHeight: document.body.scrollHeight
                    })
                """)
                if viewport_result.success:
                    info["viewport"] = viewport_result.data
                    
        except Exception as e:
            logger.error(f"Error getting page info: {e}")
            
        return info
        
    async def find_element(self, text: str = None, element_type: str = None, 
                          resource_id: str = None) -> Optional[UIElement]:
        """查找特定元素"""
        
        ui_data = await self.get_ui_hierarchy()
        if not ui_data:
            return None
            
        elements = ui_data.get("elements", [])
        
        for elem_dict in elements:
            elem = UIElement(
                element_type=elem_dict["type"],
                text=elem_dict["text"],
                bounds=tuple(elem_dict["bounds"]),
                clickable=elem_dict["clickable"],
                scrollable=elem_dict["scrollable"],
                enabled=elem_dict["enabled"],
                attributes=elem_dict["attributes"]
            )
            
            # 匹配条件
            if text and text.lower() in elem.text.lower():
                return elem
            if element_type and element_type in elem.element_type:
                return elem
            if resource_id and resource_id == elem.attributes.get('resource_id'):
                return elem
                
        return None
        
    async def find_all_elements(self, clickable: bool = None, scrollable: bool = None,
                               element_type: str = None) -> List[UIElement]:
        """查找所有符合条件的元素"""
        
        ui_data = await self.get_ui_hierarchy()
        if not ui_data:
            return []
            
        elements = ui_data.get("elements", [])
        results = []
        
        for elem_dict in elements:
            elem = UIElement(
                element_type=elem_dict["type"],
                text=elem_dict["text"],
                bounds=tuple(elem_dict["bounds"]),
                clickable=elem_dict["clickable"],
                scrollable=elem_dict["scrollable"],
                enabled=elem_dict["enabled"],
                attributes=elem_dict["attributes"]
            )
            
            # 过滤条件
            if clickable is not None and elem.clickable != clickable:
                continue
            if scrollable is not None and elem.scrollable != scrollable:
                continue
            if element_type and element_type not in elem.element_type:
                continue
                
            results.append(elem)
            
        return results