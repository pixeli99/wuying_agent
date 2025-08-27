#!/usr/bin/env python3
"""
GUI-Owl数据采集主程序 - 增强版
"""
import asyncio
import argparse
import logging
import sys
import os
from typing import List, Optional
from pathlib import Path
import json
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentbay.agent_bay import AgentBay
from gui_owl_pipeline.config import PipelineConfig, Platform, OperatingSystem
from gui_owl_pipeline.agentbay_collector import AgentBayCollector
from gui_owl_pipeline.query_generator import Query, QueryGenerator, QueryType


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志配置"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def create_sample_queries(platform: Platform, count: int = 5) -> List[Query]:
    """创建示例查询"""
    queries = []
    
    if platform == Platform.WEB:
        web_tasks = [
            ("Navigate to Google and search for AI", ["google.com", "search"], {"search_term": "AI"}),
            ("Open YouTube and search for tutorials", ["youtube.com", "search"], {"search_term": "tutorials"}),
            ("Visit GitHub and explore repositories", ["github.com", "explore"], {}),
            ("Check weather on weather.com", ["weather.com"], {}),
            ("Browse Amazon for books", ["amazon.com", "books"], {})
        ]
        
        for i, (instruction, path, slots) in enumerate(web_tasks[:count]):
            queries.append(Query(
                id=f"web_query_{i+1}",
                platform=platform,
                instruction=instruction,
                natural_instruction=f"Please {instruction.lower()}",
                path=path,
                slots=slots,
                difficulty="easy"
            ))
            
    elif platform == Platform.MOBILE:
        mobile_tasks = [
            ("Open Settings and check WiFi", ["settings", "wifi"], {}),
            ("Launch Camera and take a photo", ["camera", "capture"], {}),
            ("Open Messages and compose new message", ["messages", "compose"], {}),
            ("Navigate to Phone app and dial", ["phone", "dial"], {}),
            ("Open Gallery and view photos", ["gallery", "photos"], {})
        ]
        
        for i, (instruction, path, slots) in enumerate(mobile_tasks[:count]):
            queries.append(Query(
                id=f"mobile_query_{i+1}",
                platform=platform,
                instruction=instruction,
                natural_instruction=f"Please {instruction.lower()}",
                path=path,
                slots=slots,
                difficulty="medium"
            ))
            
    elif platform == Platform.PC:
        pc_tasks = [
            ("Open File Manager and create folder", ["file_manager", "new_folder"], {"name": "TestFolder"}),
            ("Launch Terminal and run command", ["terminal", "command"], {"command": "ls -la"}),
            ("Open Text Editor and create file", ["text_editor", "new_file"], {"name": "test.txt"}),
            ("Navigate to Desktop and organize files", ["desktop", "organize"], {}),
            ("Open System Settings", ["settings"], {})
        ]
        
        for i, (instruction, path, slots) in enumerate(pc_tasks[:count]):
            queries.append(Query(
                id=f"pc_query_{i+1}",
                platform=platform,
                instruction=instruction,
                natural_instruction=f"Please {instruction.lower()}",
                path=path,
                slots=slots,
                difficulty="medium"
            ))
            
    return queries


async def collect_single_trajectory(collector: AgentBayCollector, query: Query, 
                                   output_dir: str, save_intermediate: bool = True):
    """收集单个轨迹"""
    
    try:
        logging.info(f"Collecting trajectory for query: {query.id}")
        logging.info(f"Instruction: {query.natural_instruction}")
        
        # 收集轨迹
        trajectory = await collector.collect_trajectory_with_agentbay(query)
        
        # 记录结果
        logging.info(f"Trajectory collection completed:")
        logging.info(f"  - ID: {trajectory.id}")
        logging.info(f"  - Steps: {len(trajectory.steps)}")
        logging.info(f"  - Success: {trajectory.success}")
        if trajectory.error_message:
            logging.warning(f"  - Error: {trajectory.error_message}")
        
        # 保存轨迹
        if save_intermediate or trajectory.success:
            traj_output_dir = os.path.join(output_dir, trajectory.id)
            collector.export_agentbay_trajectory(trajectory, traj_output_dir)
            logging.info(f"  - Saved to: {traj_output_dir}")
            
        return trajectory
        
    except Exception as e:
        logging.error(f"Failed to collect trajectory for query {query.id}: {e}")
        return None


async def run_collection(args):
    """运行数据收集"""
    
    # 创建配置
    try:
        config = PipelineConfig(
            platform=Platform[args.platform.upper()],
            operating_system=OperatingSystem[args.os.upper()],
            api_key=args.api_key or os.getenv("AGENTBAY_API_KEY"),
            parallel_sessions=args.parallel,
            max_trajectory_length=args.max_steps,
            step_timeout=args.timeout,
            screenshot_interval=args.screenshot_interval,
            output_dir=args.output,
            log_level=args.log_level
        )
    except Exception as e:
        logging.error(f"Failed to create configuration: {e}")
        return 1
    
    # 创建收集器
    collector = AgentBayCollector(config)
    
    # 创建或加载查询
    if args.queries_file:
        # 从文件加载查询
        with open(args.queries_file, 'r') as f:
            queries_data = json.load(f)
            queries = [
                Query(
                    id=q["id"],
                    platform=Platform[q["platform"].upper()],
                    instruction=q["instruction"],
                    natural_instruction=q.get("natural_instruction", q["instruction"]),
                    path=q.get("path", []),
                    slots=q.get("slots", {}),
                    difficulty=q.get("difficulty", "medium")
                )
                for q in queries_data
            ]
    else:
        # 创建示例查询
        queries = create_sample_queries(config.platform, args.num_queries)
    
    if not queries:
        logging.error("No queries to process")
        return 1
    
    logging.info(f"Processing {len(queries)} queries")
    
    # 收集轨迹
    if args.batch:
        # 批量收集
        logging.info(f"Starting batch collection with {config.parallel_sessions} parallel sessions")
        trajectories = await collector.collect_batch_with_agentbay(queries)
        
        success_count = sum(1 for t in trajectories if t.success)
        logging.info(f"Batch collection completed: {success_count}/{len(trajectories)} successful")
        
    else:
        # 顺序收集
        trajectories = []
        for query in queries:
            traj = await collect_single_trajectory(
                collector, query, config.output_dir, args.save_failed
            )
            if traj:
                trajectories.append(traj)
            
            # 添加延迟避免过快请求
            if queries.index(query) < len(queries) - 1:
                await asyncio.sleep(args.delay)
    
    # 生成汇总报告
    if args.report:
        report_path = os.path.join(config.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "platform": config.platform.value,
                "os": config.operating_system.value,
                "parallel_sessions": config.parallel_sessions,
                "max_steps": config.max_trajectory_length
            },
            "queries": len(queries),
            "trajectories": len(trajectories),
            "successful": sum(1 for t in trajectories if t and t.success),
            "failed": sum(1 for t in trajectories if t and not t.success),
            "details": [
                {
                    "id": t.id,
                    "query_id": t.query.id,
                    "steps": len(t.steps),
                    "success": t.success,
                    "error": t.error_message,
                    "duration": t.end_time - t.start_time if t.end_time else None
                }
                for t in trajectories if t
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logging.info(f"Report saved to: {report_path}")
    
    return 0


async def verify_connection(api_key: str) -> bool:
    """验证AgentBay连接"""
    
    try:
        client = AgentBay(api_key=api_key)
        
        logging.info("Verifying AgentBay connection...")
        result = await client.create_session(
            image="ubuntu:22.04",
            resources={"cpu": 1, "memory": "512Mi"},
            timeout=30
        )
        
        if result.success:
            logging.info("✓ Connection successful")
            await client.delete_session(result.data.session_id)
            return True
        else:
            logging.error(f"✗ Connection failed: {result.error}")
            return False
            
    except Exception as e:
        logging.error(f"✗ Connection error: {e}")
        return False


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description="GUI-Owl Data Collection Pipeline with Wuying AgentBay SDK"
    )
    
    # 基本参数
    parser.add_argument(
        "--platform",
        choices=["web", "mobile", "pc"],
        default="web",
        help="Target platform for data collection"
    )
    parser.add_argument(
        "--os",
        choices=["ubuntu", "android", "windows", "macos"],
        default="ubuntu",
        help="Operating system"
    )
    parser.add_argument(
        "--api-key",
        help="AgentBay API key (can also use AGENTBAY_API_KEY env var)"
    )
    
    # 收集参数
    parser.add_argument(
        "--num-queries",
        type=int,
        default=3,
        help="Number of queries to generate"
    )
    parser.add_argument(
        "--queries-file",
        help="JSON file containing queries to process"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch collection (parallel)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Number of parallel sessions for batch collection"
    )
    
    # 执行参数
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum steps per trajectory"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout per step in seconds"
    )
    parser.add_argument(
        "--screenshot-interval",
        type=float,
        default=1.5,
        help="Interval between screenshots in seconds"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between sequential collections"
    )
    
    # 输出参数
    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--save-failed",
        action="store_true",
        help="Save failed trajectories"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate summary report"
    )
    
    # 日志参数
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    # 其他参数
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify connection without collecting data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be collected without actually collecting"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    
    # 检查API密钥
    api_key = args.api_key or os.getenv("AGENTBAY_API_KEY")
    if not api_key:
        logging.error("AGENTBAY_API_KEY not provided. Use --api-key or set AGENTBAY_API_KEY environment variable")
        return 1
    
    # 运行异步主函数
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 验证连接
        if args.verify:
            success = loop.run_until_complete(verify_connection(api_key))
            return 0 if success else 1
        
        # 模拟运行
        if args.dry_run:
            config = PipelineConfig(
                platform=Platform[args.platform.upper()],
                operating_system=OperatingSystem[args.os.upper()],
                api_key=api_key
            )
            queries = create_sample_queries(config.platform, args.num_queries)
            
            logging.info("Dry run - would collect the following:")
            logging.info(f"  Platform: {config.platform.value}")
            logging.info(f"  OS: {config.operating_system.value}")
            logging.info(f"  Queries: {len(queries)}")
            for q in queries:
                logging.info(f"    - {q.id}: {q.instruction}")
            return 0
        
        # 运行收集
        return loop.run_until_complete(run_collection(args))
        
    except KeyboardInterrupt:
        logging.info("Collection interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())