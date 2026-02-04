"""Central LLM Batch Broker.

Provides centralized routing for all Claude API calls with user-configurable
modes (Fast/Balanced/Economical) and call-site batch policies.

Quick Start:
    from core.llm_broker import get_broker, BatchPolicy, UserMode

    broker = get_broker()
    await broker.start()

    # Single request
    future = await broker.request(
        prompt="Analyze this text...",
        policy=BatchPolicy.PREFER_SPEED,
    )
    response = await future

    # Batch group
    async with broker.batch_group() as group:
        f1 = await broker.request(prompt1, policy=BatchPolicy.PREFER_BALANCE)
        f2 = await broker.request(prompt2, policy=BatchPolicy.PREFER_BALANCE)
        results = await asyncio.gather(f1, f2)

    await broker.stop()

Modes:
    - FAST: No batching, all calls use synchronous API
    - BALANCED: Batch calls marked PREFER_BALANCE or below (default)
    - ECONOMICAL: Aggressive batching for maximum cost savings

Policies:
    - FORCE_BATCH: Always batch (ignores Fast mode)
    - PREFER_BALANCE: Batch in Balanced/Economical modes
    - PREFER_SPEED: Batch only in Economical mode
    - REQUIRE_SYNC: Never batch, always use synchronous API
"""

from .broker import (
    LLMBroker,
    BatchGroup,
    get_broker,
    set_broker,
    reset_broker,
)
from .config import (
    BrokerConfig,
    get_broker_config,
    set_broker_config,
    reset_broker_config,
)
from .exceptions import (
    BrokerError,
    QueueOverflowError,
    BatchRequestError,
    BatchSubmissionError,
    BrokerNotStartedError,
    NestedBatchGroupError,
)
from .metrics import BrokerMetrics
from .persistence import BrokerPersistence
from .schemas import (
    BatchPolicy,
    UserMode,
    RequestState,
    LLMRequest,
    LLMResponse,
)

__all__ = [
    # Core broker
    "LLMBroker",
    "BatchGroup",
    "get_broker",
    "set_broker",
    "reset_broker",
    # Configuration
    "BrokerConfig",
    "get_broker_config",
    "set_broker_config",
    "reset_broker_config",
    # Exceptions
    "BrokerError",
    "QueueOverflowError",
    "BatchRequestError",
    "BatchSubmissionError",
    "BrokerNotStartedError",
    "NestedBatchGroupError",
    # Observability
    "BrokerMetrics",
    # Persistence
    "BrokerPersistence",
    # Schemas
    "BatchPolicy",
    "UserMode",
    "RequestState",
    "LLMRequest",
    "LLMResponse",
]
