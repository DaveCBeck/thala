"""Allow running as python -m mcp_server.web_research_tools"""

import asyncio

from .server import main

asyncio.run(main())
