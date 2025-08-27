#!/usr/bin/env python3
import asyncio
import os
from gui_owl_pipeline import DataPipeline
from gui_owl_pipeline.config import PipelineConfig, Platform, OperatingSystem

async def main():
    config = PipelineConfig(
        api_key=os.getenv("AGENTBAY_API_KEY", "your_api_key"),
        platform=Platform.PC,
        operating_system=OperatingSystem.WINDOWS,
        parallel_sessions=2,
        max_trajectory_length=15,
        step_timeout=10,
        output_dir="./output/pc_demo"
    )
    
    skills = [
        {
            "name": "Open File",
            "type": "click",
            "description": "Open a file using File menu",
            "requires_object": False
        },
        {
            "name": "Save Document",
            "type": "click",
            "description": "Save the current document",
            "requires_object": False
        },
        {
            "name": "Type Text",
            "type": "type",
            "description": "Type text in the editor",
            "requires_object": False
        },
        {
            "name": "Select All",
            "type": "key_press",
            "description": "Select all content (Ctrl+A)",
            "requires_object": False
        },
        {
            "name": "Copy",
            "type": "key_press",
            "description": "Copy selected content (Ctrl+C)",
            "requires_object": True
        },
        {
            "name": "Paste",
            "type": "key_press",
            "description": "Paste from clipboard (Ctrl+V)",
            "requires_object": False
        },
        {
            "name": "Format Bold",
            "type": "click",
            "description": "Make selected text bold",
            "requires_object": True
        },
        {
            "name": "Insert Image",
            "type": "drag",
            "description": "Insert an image by dragging",
            "requires_object": True
        },
        {
            "name": "Resize Window",
            "type": "drag",
            "description": "Resize application window",
            "requires_object": False
        },
        {
            "name": "Close Application",
            "type": "click",
            "description": "Close the application",
            "requires_object": False
        }
    ]
    
    pipeline = DataPipeline(config)
    
    print("Initializing GUI-Owl Data Pipeline for PC...")
    await pipeline.initialize()
    
    print("\nRunning PC data generation pipeline...")
    result = await pipeline.run_pc_pipeline(
        software_name="Text Editor",
        skills=skills,
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