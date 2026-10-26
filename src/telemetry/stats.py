"""In-process request counters and cost ledger (Prometheus metrics arrive in Phase 3)."""
from pydantic import BaseModel


class GatewayStats(BaseModel):
    total_requests: int = 0
    cache_hits: int = 0
    guardrail_blocks: int = 0
    local_routed_calls: int = 0
    cloud_routed_calls: int = 0
    failover_saves: int = 0
    upstream_errors: int = 0
    money_saved_usd: float = 0.0

    def add_savings(self, amount_usd: float) -> None:
        self.money_saved_usd = round(self.money_saved_usd + max(0.0, amount_usd), 6)
