"""
AAG DLP Client - HTTP client for DLP API integration.

Allows other services (Agent Profiler, Deception Orchestrator, etc.)
to call the DLP inspection endpoints.
"""

import asyncio
import httpx
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class DLPAction(str, Enum):
    MASKING = "MASKING"
    REDACTION = "REDACTION"
    BLOCK = "BLOCK"
    HONEYTOKEN_INJECTION = "HONEYTOKEN_INJECTION"


class DLPViolationType(str, Enum):
    # Brazilian PII
    CPF = "CPF"
    CNPJ = "CNPJ"
    PHONE_BR = "PHONE_BR"
    EMAIL = "EMAIL"
    ADDRESS_BR = "ADDRESS_BR"
    RG = "RG"
    CNH = "CNH"
    PIS = "PIS"
    TITULO_ELEITOR = "TITULO_ELEITOR"
    CERTIDAO_NASCIMENTO = "CERTIDAO_NASCIMENTO"
    CERTIDAO_CASAMENTO = "CERTIDAO_CASAMENTO"
    CEP = "CEP"

    # International PII
    SSN_US = "SSN_US"
    PASSPORT = "PASSPORT"
    DRIVERS_LICENSE_US = "DRIVERS_LICENSE_US"
    PHONE_US = "PHONE_US"
    ADDRESS_US = "ADDRESS_US"
    IBAN = "IBAN"

    # Secrets & Credentials
    AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
    AWS_SECRET_KEY = "AWS_SECRET_KEY"
    GCP_API_KEY = "GCP_API_KEY"
    GITHUB_TOKEN = "GITHUB_TOKEN"
    GITHUB_PAT = "GITHUB_PAT"
    SLACK_TOKEN = "SLACK_TOKEN"
    DISCORD_TOKEN = "DISCORD_TOKEN"
    JWT_TOKEN = "JWT_TOKEN"
    DB_CONNECTION_STRING = "DB_CONNECTION_STRING"
    API_KEY_GENERIC = "API_KEY_GENERIC"
    PRIVATE_KEY = "PRIVATE_KEY"
    SSH_PRIVATE_KEY = "SSH_PRIVATE_KEY"
    PASSWORD_IN_TEXT = "PASSWORD_IN_TEXT"
    BEARER_TOKEN = "BEARER_TOKEN"

    # Financial
    CREDIT_CARD = "CREDIT_CARD"
    CREDIT_CARD_TRACK = "CREDIT_CARD_TRACK"
    BANK_ACCOUNT_BR = "BANK_ACCOUNT_BR"
    BANK_ACCOUNT_US = "BANK_ACCOUNT_US"
    SWIFT_BIC = "SWIFT_BIC"
    CRYPTO_WALLET_BTC = "CRYPTO_WALLET_BTC"
    CRYPTO_WALLET_ETH = "CRYPTO_WALLET_ETH"

    # Network/Infrastructure
    IP_ADDRESS = "IP_ADDRESS"
    MAC_ADDRESS = "MAC_ADDRESS"
    UUID = "UUID"
    INTERNAL_HOSTNAME = "INTERNAL_HOSTNAME"


@dataclass
class DLPViolation:
    violation_type: DLPViolationType
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str
    masked_value: str


@dataclass
class DLPInspectionResult:
    original_text_length: int
    processed_text: str
    violations_count: int
    violations: List[DLPViolation]
    action_taken: DLPAction
    blocked: bool
    metadata: Dict[str, Any]


class DLPClient:
    """
    Async HTTP client for AAG DLP API.

    Usage:
        client = DLPClient(base_url="http://localhost:8003")
        result = await client.inspect("Meu CPF é 123.456.789-09", direction="outbound")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8003",
        timeout: float = 30.0,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _ensure_client(self):
        if self._client is None:
            await self.__aenter__()

    def _parse_result(self, data: Dict[str, Any]) -> DLPInspectionResult:
        violations = [
            DLPViolation(
                violation_type=DLPViolationType(v["violation_type"]),
                matched_text=v["matched_text"],
                start_pos=v["start_pos"],
                end_pos=v["end_pos"],
                confidence=v["confidence"],
                context=v["context"],
                masked_value=v["masked_value"]
            )
            for v in data["violations"]
        ]
        return DLPInspectionResult(
            original_text_length=data["original_text_length"],
            processed_text=data["processed_text"],
            violations_count=data["violations_count"],
            violations=violations,
            action_taken=DLPAction(data["action_taken"]),
            blocked=data["blocked"],
            metadata=data["metadata"]
        )

    async def inspect(
        self,
        text: str,
        direction: str = "outbound",
        session_id: str = "",
        profile: str = "balanced",
        honeytoken_enabled: bool = False
    ) -> DLPInspectionResult:
        """
        Inspect text for sensitive data.

        Args:
            text: Text to inspect
            direction: "inbound" or "outbound"
            session_id: Session identifier for context
            profile: DLP profile (strict, balanced, monitor, lgpd, pci-dss)
            honeytoken_enabled: Enable honeytoken injection

        Returns:
            DLPInspectionResult with violations and processed text
        """
        await self._ensure_client()

        payload = {
            "text": text,
            "direction": direction,
            "session_id": session_id,
            "profile": profile,
            "honeytoken_enabled": honeytoken_enabled
        }

        response = await self._client.post("/dlp/inspect", json=payload)
        response.raise_for_status()
        return self._parse_result(response.json())

    async def inspect_batch(
        self,
        texts: List[str],
        direction: str = "outbound",
        session_id: str = "",
        profile: str = "balanced",
        honeytoken_enabled: bool = False
    ) -> List[DLPInspectionResult]:
        """Inspect multiple texts in batch."""
        await self._ensure_client()

        payload = [
            {
                "text": text,
                "direction": direction,
                "session_id": session_id,
                "profile": profile,
                "honeytoken_enabled": honeytoken_enabled
            }
            for text in texts
        ]

        response = await self._client.post("/dlp/inspect/batch", json=payload)
        response.raise_for_status()
        data = response.json()
        return [self._parse_result(r) for r in data["results"]]

    async def health_check(self) -> Dict[str, Any]:
        """Check DLP service health."""
        await self._ensure_client()
        response = await self._client.get("/dlp/health")
        response.raise_for_status()
        return response.json()

    async def list_rules(self) -> Dict[str, Any]:
        """List all loaded DLP rules."""
        await self._ensure_client()
        response = await self._client.get("/dlp/rules")
        response.raise_for_status()
        return response.json()

    async def configure(
        self,
        profile: str = "balanced",
        honeytoken_enabled: bool = False,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Configure DLP engine."""
        await self._ensure_client()

        payload = {
            "profile": profile,
            "honeytoken_enabled": honeytoken_enabled,
            "custom_rules": custom_rules or []
        }

        response = await self._client.post("/dlp/config", json=payload)
        response.raise_for_status()
        return response.json()

    async def generate_honeytoken(self, violation_type: str) -> Dict[str, Any]:
        """Generate a honeytoken for a specific violation type."""
        await self._ensure_client()

        payload = {"violation_type": violation_type}
        response = await self._client.post("/dlp/honeytoken/generate", json=payload)
        response.raise_for_status()
        return response.json()


class SyncDLPClient:
    """Synchronous wrapper for DLPClient."""

    def __init__(
        self,
        base_url: str = "http://localhost:8003",
        timeout: float = 30.0,
        api_key: Optional[str] = None
    ):
        self._async_client = DLPClient(base_url, timeout, api_key)

    def _run(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running in async context - create new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    def inspect(
        self,
        text: str,
        direction: str = "outbound",
        session_id: str = "",
        profile: str = "balanced",
        honeytoken_enabled: bool = False
    ) -> DLPInspectionResult:
        return self._run(self._async_client.inspect(
            text, direction, session_id, profile, honeytoken_enabled
        ))

    def inspect_batch(
        self,
        texts: List[str],
        direction: str = "outbound",
        session_id: str = "",
        profile: str = "balanced",
        honeytoken_enabled: bool = False
    ) -> List[DLPInspectionResult]:
        return self._run(self._async_client.inspect_batch(
            texts, direction, session_id, profile, honeytoken_enabled
        ))

    def health_check(self) -> Dict[str, Any]:
        return self._run(self._async_client.health_check())

    def list_rules(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_rules())

    def configure(
        self,
        profile: str = "balanced",
        honeytoken_enabled: bool = False,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        return self._run(self._async_client.configure(
            profile, honeytoken_enabled, custom_rules
        ))

    def generate_honeytoken(self, violation_type: str) -> Dict[str, Any]:
        return self._run(self._async_client.generate_honeytoken(violation_type))


# Convenience functions for simple usage
async def quick_inspect(
    text: str,
    base_url: str = "http://localhost:8003",
    direction: str = "outbound",
    session_id: str = "",
    profile: str = "balanced"
) -> DLPInspectionResult:
    """Quick one-off inspection without managing client lifecycle."""
    async with DLPClient(base_url) as client:
        return await client.inspect(text, direction, session_id, profile)


def sync_quick_inspect(
    text: str,
    base_url: str = "http://localhost:8003",
    direction: str = "outbound",
    session_id: str = "",
    profile: str = "balanced"
) -> DLPInspectionResult:
    """Synchronous quick inspection."""
    client = SyncDLPClient(base_url)
    return client.inspect(text, direction, session_id, profile)


if __name__ == "__main__":
    # Example usage
    async def main():
        async with DLPClient("http://localhost:8003") as client:
            # Health check
            health = await client.health_check()
            print(f"Health: {health}")

            # Inspect text
            result = await client.inspect(
                "Meu CPF é 123.456.789-09 e meu cartão é 4532 0151 1283 0366",
                direction="outbound",
                session_id="test-session"
            )
            print(f"Processed: {result.processed_text}")
            print(f"Violations: {result.violations_count}")
            for v in result.violations:
                print(f"  - {v.violation_type}: {v.matched_text} -> {v.masked_value}")

    asyncio.run(main())