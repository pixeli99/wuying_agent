#!/usr/bin/env python3
import asyncio
import os
from gui_owl_pipeline import DataPipeline
from gui_owl_pipeline.config import PipelineConfig, Platform, OperatingSystem
from gui_owl_pipeline.query_generator import Page, PageTransition

async def main():
    config = PipelineConfig(
        api_key=os.getenv("AGENTBAY_API_KEY", "your_api_key"),
        platform=Platform.MOBILE,
        operating_system=OperatingSystem.ANDROID,
        parallel_sessions=3,
        max_trajectory_length=20,
        step_timeout=10,
        output_dir="./output/mobile_demo"
    )
    
    pages = [
        Page(
            id="home",
            name="Home Screen",
            description="Main home screen with app icons",
            elements=[
                {"type": "icon", "name": "Settings", "position": {"x": 100, "y": 200}},
                {"type": "icon", "name": "Camera", "position": {"x": 200, "y": 200}},
                {"type": "icon", "name": "Browser", "position": {"x": 300, "y": 200}}
            ]
        ),
        Page(
            id="settings",
            name="Settings",
            description="System settings page with various options",
            elements=[
                {"type": "option", "name": "WiFi", "position": {"x": 50, "y": 100}},
                {"type": "option", "name": "Bluetooth", "position": {"x": 50, "y": 150}},
                {"type": "option", "name": "Display", "position": {"x": 50, "y": 200}}
            ]
        ),
        Page(
            id="wifi_settings",
            name="WiFi Settings",
            description="WiFi configuration page",
            elements=[
                {"type": "switch", "name": "WiFi Toggle", "position": {"x": 300, "y": 50}},
                {"type": "list", "name": "Available Networks", "position": {"x": 50, "y": 150}}
            ]
        ),
        Page(
            id="network_details",
            name="Network Details",
            description="Selected network configuration form",
            elements=[
                {"type": "input", "name": "Password", "position": {"x": 50, "y": 200}},
                {"type": "button", "name": "Connect", "position": {"x": 150, "y": 300}}
            ]
        )
    ]
    
    transitions = [
        PageTransition(
            from_page="home",
            to_page="settings",
            action="click_settings"
        ),
        PageTransition(
            from_page="settings",
            to_page="wifi_settings",
            action="click_wifi"
        ),
        PageTransition(
            from_page="wifi_settings",
            to_page="network_details",
            action="select_network"
        ),
        PageTransition(
            from_page="network_details",
            to_page="wifi_settings",
            action="connect_network",
            condition="password_correct"
        )
    ]
    
    pipeline = DataPipeline(config)
    
    print("Initializing GUI-Owl Data Pipeline for Mobile...")
    await pipeline.initialize()
    
    print("\nRunning mobile data generation pipeline...")
    result = await pipeline.run_mobile_pipeline(
        app_name="Android System",
        pages=pages,
        transitions=transitions,
        query_count=5
    )
    
    print(f"\n=== Pipeline Results ===")
    print(f"Total Queries: {result.total_queries}")
    print(f"Total Trajectories: {result.total_trajectories}")
    print(f"Successful: {result.successful_trajectories}")
    print(f"Failed: {result.failed_trajectories}")
    print(f"High Quality: {result.high_quality_trajectories}")
    print(f"Guidance Generated: {result.guidance_generated}")
    print(f"Duration: {result.duration:.2f} seconds")
    
    print("\nGenerating downstream data...")
    downstream_data = await pipeline.generate_downstream_data()
    print(f"Grounding Data: {len(downstream_data['grounding'])} samples")
    print(f"Planning Data: {len(downstream_data['planning'])} samples")
    print(f"Action Semantic Data: {len(downstream_data['action_semantics'])} samples")
    
    print("\nExporting pipeline data...")
    pipeline.export_pipeline_data(config.output_dir)
    print(f"Data exported to {config.output_dir}")
    
    print("\nPipeline Statistics:")
    stats = pipeline.get_statistics()
    for category, data in stats.items():
        print(f"\n{category.upper()}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    print("\nCleaning up...")
    await pipeline.cleanup()
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())