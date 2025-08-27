from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import os

class Platform(Enum):
    MOBILE = "mobile"
    PC = "pc" 
    WEB = "web"

class OperatingSystem(Enum):
    ANDROID = "android"
    UBUNTU = "ubuntu"
    MACOS = "macos" 
    WINDOWS = "windows"

class ActionType(Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SWIPE = "swipe"
    DOUBLE_CLICK = "double_click"
    DRAG = "drag"
    KEY_PRESS = "key_press"

class CriticLabel(Enum):
    GOOD = "GOOD"
    NEUTRAL = "NEUTRAL"
    HARMFUL = "HARMFUL"

@dataclass
class PipelineConfig:
    platform: Platform
    operating_system: OperatingSystem
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("AGENTBAY_API_KEY"))
    parallel_sessions: int = 5
    max_trajectory_length: int = 50
    step_timeout: int = 30
    screenshot_interval: float = 1.0
    
    llm_model: str = "gpt-4o"
    vlm_model: str = "gpt-4o"
    critic_threshold: float = 0.7
    
    chunk_size: int = 1024 * 1024
    max_retries: int = 3
    retry_delay: float = 1.0
    
    output_dir: str = "./output"
    log_level: str = "INFO"
    
    # Session configuration
    session_resources: Dict[str, Any] = field(default_factory=dict)
    session_timeout_multiplier: int = 2
    
    # Browser specific
    browser_headless: bool = False
    browser_viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1920, "height": 1080})
    
    # Mobile specific
    android_device_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate API key
        if not self.api_key:
            raise ValueError("AGENTBAY_API_KEY must be set either in config or environment")
        
        # Set default resources based on platform
        if not self.session_resources:
            if self.platform == Platform.MOBILE:
                self.session_resources = {"cpu": 2, "memory": "4Gi"}
            elif self.platform == Platform.PC:
                self.session_resources = {"cpu": 4, "memory": "8Gi"}
            else:  # WEB
                self.session_resources = {"cpu": 2, "memory": "4Gi"}
        
        # Validate configuration
        if self.parallel_sessions < 1:
            raise ValueError("parallel_sessions must be at least 1")
        if self.max_trajectory_length < 1:
            raise ValueError("max_trajectory_length must be at least 1")
        if self.step_timeout < 1:
            raise ValueError("step_timeout must be at least 1 second")
        if self.screenshot_interval < 0:
            raise ValueError("screenshot_interval must be non-negative")
        if self.critic_threshold < 0 or self.critic_threshold > 1:
            raise ValueError("critic_threshold must be between 0 and 1")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    @classmethod
    def from_env(cls, platform: Platform = Platform.WEB, operating_system: OperatingSystem = OperatingSystem.UBUNTU) -> "PipelineConfig":
        """Create config from environment variables"""
        return cls(
            platform=platform,
            operating_system=operating_system,
            api_key=os.getenv("AGENTBAY_API_KEY"),
            parallel_sessions=int(os.getenv("AGENTBAY_PARALLEL_SESSIONS", "5")),
            max_trajectory_length=int(os.getenv("AGENTBAY_MAX_TRAJECTORY", "50")),
            step_timeout=int(os.getenv("AGENTBAY_STEP_TIMEOUT", "30")),
            output_dir=os.getenv("AGENTBAY_OUTPUT_DIR", "./output")
        )