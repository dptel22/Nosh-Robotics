import asyncio
from contextlib import AsyncExitStack
import sys
import webbrowser
from fastmcp import Client
from colab_mcp.websocket_server import ColabWebSocketServer
from colab_mcp.session import ColabTransport

sys.stdout.reconfigure(line_buffering=True)

async def run_bridge(timeout_seconds=600):
    async with AsyncExitStack() as stack:
        print("Starting Colab WebSocket Server...")
        wss = await stack.enter_async_context(ColabWebSocketServer())
        token = wss.token
        port = wss.port
        url = f"https://colab.research.google.com/github/dptel22/Nosh-Robotics/blob/main/Task_1.ipynb#mcpProxyToken={token}&mcpProxyPort={port}"
        
        print("\n" + "=" * 70)
        print("COLAB MCP BRIDGE ACTIVE")
        print("=" * 70)
        print(f"Colab Connection URL:\n{url}\n")
        print("Please click/open this URL in your browser with Colab.")
        print(f"Listening on port {port}... (Waiting up to {timeout_seconds} seconds)")
        print("=" * 70 + "\n")
        
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            
        try:
            await asyncio.wait_for(wss.connection_live.wait(), timeout=timeout_seconds)
            print("\n>>> SUCCESS: Colab frontend connected via WebSocket! <<<\n")
            
            client = await stack.enter_async_context(Client(ColabTransport(wss)))
            tools = await client.list_tools()
            print("--- Available Tools from Colab Frontend ---")
            for t in tools:
                print(f"Tool: {t.name}")
                print(f"  Description: {t.description}")
                print(f"  Schema: {t.parameters}\n")
                
            # Keep alive and handle requests
            print("Colab MCP session is live and ready.")
            while wss.connection_live.is_set():
                await asyncio.sleep(2)
        except asyncio.TimeoutError:
            print(f"Timed out after {timeout_seconds}s waiting for Colab frontend connection.")
        except Exception as e:
            print(f"Bridge encountered an error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    timeout = 600
    if len(sys.argv) > 1:
        try:
            timeout = int(sys.argv[1])
        except ValueError:
            pass
    asyncio.run(run_bridge(timeout))
