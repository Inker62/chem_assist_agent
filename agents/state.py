import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage


def _last_write(old, new):
    """并行 fan-out 时多个节点写入同一 key，取最后的覆盖"""
    return new


class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: Annotated[str | None, _last_write]
    round_count: int
    both_active: bool  # True when supervisor decided "both" and lit hasn't run yet
