from typing import Any, Dict, TypedDict
from typing_extensions import TypedDict

class AgentState(TypedDict):
    state: str
    results: Dict[str, Any]