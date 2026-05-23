from collections import defaultdict
from typing import Dict, Any, List
import asyncio

jobs: Dict[str, Dict[str, Any]] = {}
job_subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
stac_cache: Dict[str, Dict[str, Any]] = {}