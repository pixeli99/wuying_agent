#!/usr/bin/env python3
"""
真实的GUI-Owl数据采集示例 - 使用Wuying AgentBay SDK
"""
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentbay.agent_bay import AgentBay
from gui_owl_pipeline.agentbay_collector import AgentBayCollector
from gui_owl_pipeline.query_generator import Query
from gui_owl_pipeline.config import PipelineConfig, Platform, OperatingSystem

async def test_web_data_collection():
    """测试Web浏览器数据采集"""
    
    print("=== Web Browser Data Collection Demo ===\n")
    
    # 配置
    config = PipelineConfig(
        platform=Platform.WEB,
        operating_system=OperatingSystem.UBUNTU,
        api_key=os.getenv("AGENTBAY_API_KEY"),
        parallel_sessions=1,
        max_trajectory_length=10,
        step_timeout=30,
        screenshot_interval=2.0,
        output_dir="./output/web_real_demo"
    )
    
    # 创建收集器
    collector = AgentBayCollector(config)
    
    # 创建测试Query
    query = Query(
        id="web_search_01",
        platform=Platform.WEB,
        instruction="Navigate to Google and search for 'OpenAI'",
        natural_instruction="Please go to Google and search for information about OpenAI",
        path=["google.com", "search", "results"],
        slots={"search_term": "OpenAI"},
        difficulty="easy"
    )
    
    try:
        # 收集轨迹
        print(f"Collecting trajectory for: {query.natural_instruction}")
        trajectory = await collector.collect_trajectory_with_agentbay(query)
        
        print(f"\nTrajectory collected:")
        print(f"  - ID: {trajectory.id}")
        print(f"  - Session: {trajectory.session_id}")
        print(f"  - Steps collected: {len(trajectory.steps)}")
        
        # 导出数据
        output_dir = os.path.join(config.output_dir, trajectory.id)
        collector.export_agentbay_trajectory(trajectory, output_dir)
        print(f"  - Data exported to: {output_dir}")
        
        # 显示步骤详情
        print("\nStep details:")
        for step in trajectory.steps:
            print(f"  Step {step.index}:")
            print(f"    - Action: {step.action.get('type')}")
            print(f"    - Success: {step.execution_result.get('success', False)}")
            if step.browser_state:
                print(f"    - URL: {step.browser_state.get('url', 'N/A')}")
                print(f"    - Title: {step.browser_state.get('title', 'N/A')}")
                
    except Exception as e:
        print(f"Error during collection: {e}")
        
async def test_mobile_data_collection():
    """测试Android移动端数据采集"""
    
    print("=== Android Mobile Data Collection Demo ===\n")
    
    # 配置
    config = PipelineConfig(
        platform=Platform.MOBILE,
        operating_system=OperatingSystem.ANDROID,
        api_key=os.getenv("AGENTBAY_API_KEY"),
        parallel_sessions=1,
        max_trajectory_length=15,
        step_timeout=30,
        screenshot_interval=1.5,
        output_dir="./output/mobile_real_demo"
    )
    
    # 创建收集器
    collector = AgentBayCollector(config)
    
    # 创建测试Query
    query = Query(
        id="android_settings_01",
        platform=Platform.MOBILE,
        instruction="Open Settings and navigate to WiFi settings",
        natural_instruction="Please open the Settings app and go to WiFi configuration",
        path=["home", "settings", "wifi"],
        slots={},
        difficulty="easy"
    )
    
    try:
        # 收集轨迹
        print(f"Collecting trajectory for: {query.natural_instruction}")
        trajectory = await collector.collect_trajectory_with_agentbay(query)
        
        print(f"\nTrajectory collected:")
        print(f"  - ID: {trajectory.id}")
        print(f"  - Session: {trajectory.session_id}")
        print(f"  - Steps collected: {len(trajectory.steps)}")
        
        # 导出数据
        output_dir = os.path.join(config.output_dir, trajectory.id)
        collector.export_agentbay_trajectory(trajectory, output_dir)
        print(f"  - Data exported to: {output_dir}")
        
        # 显示UI元素信息
        print("\nUI elements detected:")
        for step in trajectory.steps:
            if step.ui_data:
                elements = step.ui_data.get("elements", [])
                print(f"  Step {step.index}: {len(elements)} clickable elements found")
                
    except Exception as e:
        print(f"Error during collection: {e}")
        
async def test_pc_data_collection():
    """测试PC桌面端数据采集"""
    
    print("=== PC Desktop Data Collection Demo ===\n")
    
    # 配置
    config = PipelineConfig(
        platform=Platform.PC,
        operating_system=OperatingSystem.UBUNTU,
        api_key=os.getenv("AGENTBAY_API_KEY"),
        parallel_sessions=1,
        max_trajectory_length=10,
        step_timeout=30,
        screenshot_interval=2.0,
        output_dir="./output/pc_real_demo"
    )
    
    # 创建收集器
    collector = AgentBayCollector(config)
    
    # 创建测试Query
    query = Query(
        id="desktop_file_01",
        platform=Platform.PC,
        instruction="Open file manager and create a new folder",
        natural_instruction="Please open the file manager and create a new folder called 'TestFolder'",
        path=["desktop", "file_manager", "new_folder"],
        slots={"folder_name": "TestFolder"},
        difficulty="medium"
    )
    
    try:
        # 收集轨迹
        print(f"Collecting trajectory for: {query.natural_instruction}")
        trajectory = await collector.collect_trajectory_with_agentbay(query)
        
        print(f"\nTrajectory collected:")
        print(f"  - ID: {trajectory.id}")
        print(f"  - Session: {trajectory.session_id}")
        print(f"  - Steps collected: {len(trajectory.steps)}")
        
        # 导出数据
        output_dir = os.path.join(config.output_dir, trajectory.id)
        collector.export_agentbay_trajectory(trajectory, output_dir)
        print(f"  - Data exported to: {output_dir}")
        
        # 显示窗口信息
        print("\nWindow information:")
        for step in trajectory.steps:
            if step.window_info:
                windows = step.window_info.get("windows", [])
                print(f"  Step {step.index}: {len(windows)} windows detected")
                
    except Exception as e:
        print(f"Error during collection: {e}")
        
async def test_batch_collection():
    """测试批量数据采集"""
    
    print("=== Batch Data Collection Demo ===\n")
    
    # 配置
    config = PipelineConfig(
        platform=Platform.WEB,
        operating_system=OperatingSystem.UBUNTU,
        api_key=os.getenv("AGENTBAY_API_KEY"),
        parallel_sessions=3,
        max_trajectory_length=5,
        step_timeout=30,
        screenshot_interval=1.0,
        output_dir="./output/batch_demo"
    )
    
    # 创建收集器
    collector = AgentBayCollector(config)
    
    # 创建多个Query
    queries = [
        Query(
            id=f"batch_query_{i}",
            platform=Platform.WEB,
            instruction=f"Task {i}: Navigate to website",
            natural_instruction=f"Please navigate to example website {i}",
            path=["home", "page"],
            slots={},
            difficulty="easy"
        )
        for i in range(3)
    ]
    
    try:
        # 批量收集
        print(f"Collecting {len(queries)} trajectories in parallel...")
        trajectories = await collector.collect_batch_with_agentbay(queries)
        
        print(f"\nBatch collection completed:")
        print(f"  - Total queries: {len(queries)}")
        print(f"  - Successful collections: {len(trajectories)}")
        
        # 导出所有轨迹
        for traj in trajectories:
            output_dir = os.path.join(config.output_dir, traj.id)
            collector.export_agentbay_trajectory(traj, output_dir)
            print(f"  - Exported: {traj.id}")
            
    except Exception as e:
        print(f"Error during batch collection: {e}")
        
async def verify_agentbay_connection():
    """验证AgentBay连接"""
    
    print("=== Verifying AgentBay Connection ===\n")
    
    api_key = os.getenv("AGENTBAY_API_KEY")
    if not api_key:
        print("ERROR: AGENTBAY_API_KEY environment variable not set")
        print("Please set it with: export AGENTBAY_API_KEY='your_key'")
        return False
        
    try:
        client = AgentBay(api_key=api_key)
        
        # 尝试创建一个测试会话
        print("Creating test session...")
        result = await client.create_session(
            image="ubuntu:22.04",
            resources={"cpu": 1, "memory": "1Gi"},
            timeout=60
        )
        
        if result.success:
            print(f"✓ Connection successful!")
            print(f"  - Session ID: {result.data.session_id}")
            
            # 清理测试会话
            await client.delete_session(result.data.session_id)
            print("  - Test session cleaned up")
            return True
        else:
            print(f"✗ Connection failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False
        
async def main():
    """主函数"""
    
    print("=" * 60)
    print("GUI-Owl Data Collection with Wuying AgentBay SDK")
    print("=" * 60)
    print()
    
    # 验证连接
    if not await verify_agentbay_connection():
        print("\nPlease configure your AgentBay API key and try again.")
        return
        
    print("\nSelect demo to run:")
    print("1. Web Browser Data Collection")
    print("2. Android Mobile Data Collection")
    print("3. PC Desktop Data Collection")
    print("4. Batch Collection Demo")
    print("5. Run All Demos")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        await test_web_data_collection()
    elif choice == "2":
        await test_mobile_data_collection()
    elif choice == "3":
        await test_pc_data_collection()
    elif choice == "4":
        await test_batch_collection()
    elif choice == "5":
        await test_web_data_collection()
        print("\n" + "=" * 60 + "\n")
        await test_mobile_data_collection()
        print("\n" + "=" * 60 + "\n")
        await test_pc_data_collection()
        print("\n" + "=" * 60 + "\n")
        await test_batch_collection()
    else:
        print("Invalid choice")
        
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())