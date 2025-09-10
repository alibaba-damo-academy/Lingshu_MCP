#!/usr/bin/env python3
"""
Lingshu 32B FastMCP Client Test Script

This script demonstrates how to interact with the Lingshu FastMCP server,
simulating the usage scenario of dHealth Intelligence.
"""

import asyncio
import base64
import json
import httpx
from datetime import datetime
from typing import Dict, Any
import os
import argparse
from fastmcp import Client
from openai import AsyncOpenAI

llm_client = AsyncOpenAI(
    base_url=os.environ.get("LLM_SERVER_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("LLM_SERVER_API", "api_key")  
)
model = os.environ.get("LLM_MODEL", "qwen3-235b-a22b-instruct-2507")
async def query_mcp_tool( tool_name: str, params: dict):
    """
    调用MCP工具的统一入口
    :param tool_name: 工具名称
    :param params: 工具参数
    :return: 工具执行结果
    """
    async with Client("http://127.0.0.1:4200/lingshu") as client:
        return await client.call_tool(tool_name, params)

async def test_image_analysis(mcp_server_url):
    """Test medical image analysis"""
    print("\n🖼️  Testing medical image analysis...")
    
    # Create a test image (in actual use, this should be a real medical image)
    print("Note: This test requires a real medical image file")
    
    # If you have a test image file, uncomment the following and provide the correct path
    test_image_path = "./lung.jpeg"
    async with Client(mcp_server_url) as mcp_client:
        if os.path.exists(test_image_path):
            
            result = await query_mcp_tool(
                "analyze_medical_image",
                params= {
                    "image_path": test_image_path,
                    "analysis_type": "radiology",
                    "patient_context": "55-year-old male patient, 20-year smoking history, abnormality found in lung during physical examination",
                    "language": "en"
                }
            )
            # import pdb; pdb.set_trace()
            print(result)
                    
        else:
            print("⚠️  Test image file does not exist, skipping image analysis test")

async def main(mcp_server_url: str):
    """
    Implement chat functionality with tool call support
    1. Connect to local vLLM service
    2. Get available tool list and convert to OpenAI function call format
    3. Call appropriate tools based on user questions
    4. Integrate tool results to generate final response
    """
    
    async with Client(mcp_server_url) as mcp_client:
        # Dynamically get the list of tools provided by the MCP service
        tools = await mcp_client.list_tools()
        # Convert MCP tool schemas to OpenAI function call format
        tool_schemas = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": tool.inputSchema.get("type", "object"),
                    "properties": {
                        prop_name: prop_def 
                        for prop_name, prop_def in tool.inputSchema["properties"].items()
                    },
                    "required": tool.inputSchema.get("required", [])
                }
            }
        } for tool in tools]
        # User query example
        print("--" * 20)
        print("Available tools:")
        for tool in tools:
            print(f"- {tool}\n")

        print("--" * 20)    
        user_query = "How to evaluate lung nodules in CT images? What imaging features should be noted?"

        # First call to the model, allowing it to decide if tool calls are needed
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_query}],
            tools=tool_schemas,
            tool_choice="auto"  # Let the model automatically choose tools
        )
        
        # Handle tool call requests
        message = response.choices[0].message
        print(message.tool_calls)

        if message.tool_calls:
            print("Tool call request detected:")

            # Execute all tools requested by the model in order
            for call in message.tool_calls:
                print(f"Executing {call.function.name}...")
                # Call MCP tool and get results
                result = await query_mcp_tool(
                    call.function.name,
                    eval(call.function.arguments)  # Convert argument string to dictionary
                )
                print(f"Tool returned: {result}")
        else:
            # If the model decides no tools are needed, return the model's reply directly
            print("Direct reply:", message.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lingshu 32B FastMCP Client Test Script")
    parser.add_argument("--mcp-url", default="http://127.0.0.1:4200/lingshu",
                        help="MCP server URL (default: http://127.0.0.1:4200/lingshu)")
    args = parser.parse_args()
    
    # asyncio.run(main(args.mcp_url))
    asyncio.run(test_image_analysis(args.mcp_url))
