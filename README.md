# Lingshu FastMCP Medical AI Service

This project implements a FastMCP server for the Lingshu medical AI model and a corresponding client for testing and integration.

## Components

1. `mcp_server_lingshu.py`: FastMCP server wrapping the Lingshu model
2. `mcp_client_lingshu.py`: Test client demonstrating interaction with the Lingshu FastMCP server

## Server Features

- Medical image analysis
- Structured medical report generation
- Medical Q&A

## Prerequisites

- FastMCP framework
- OpenAI API compatible LLM server (e.g., vLLM)
- Required Python packages (install via `pip install -r requirements.txt`)

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

## Usage

### Use vLLM to serve the Lingshu Model
```bash
vllm serve lingshu-medical-mllm/Lingshu-7B  --dtype float16 --api_key api_key --port 8000  --max-model-len 32768
```
### Wrap the server with FastMCP
```python
export LINGSHU_SERVER_URL="http://localhost:8000/v1" 
export LINGSHU_SERVER_API="api_key"
export LINGSHU_MODEL="lingshu-medical-mllm/Lingshu-7B" # the above config depends on your vllm server config
python mcp_server_lingshu.py --host 127.0.0.1 --port 4200 --path /lingshu --log-level info
```
### Try connect Lingshu with MCP
```python
export LLM_SERVER_URL="xxx"
export LLM_SERVER_API="xxx"
export LLM_MODEL="xxx" ## this is your own model
python mcp_client_lingshu.py  --mcp-url http://127.0.0.1:4200/lingshu # the mcp-url should depend on the mcp server you deployed in the last step
```