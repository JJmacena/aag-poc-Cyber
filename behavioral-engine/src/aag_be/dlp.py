"""
AAG PoC - Data Loss Prevention (DLP) Module

Robust DLP engine for detecting and preventing exfiltration of sensitive data
in LLM-based applications. Supports PII, secrets, and financial data detection
with configurable mitigation policies.

LGPD/GDPR compliant - designed for Brazilian and international data protection.
"""

import re
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DLPAction(Enum):
    """DLP mitigation action levels."""
    MASKING = "MASKING"           # Replace with placeholder [TYPE_REDACTED]
    REDACTION = "REDACTION"       # Remove completely (empty string)
    BLOCK = "BLOCK"               # Stop execution, raise violation
    HONEYTOKEN_INJECTION = "HONEYTOKEN_INJECTION"  # Inject fake monitored data


class DLPViolationType(Enum):
    """Types of DLP violations detected."""
    # PII - Brazilian
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

    # PII - International
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

    # Healthcare (HIPAA)
    MEDICAL_RECORD_NUMBER = "MEDICAL_RECORD_NUMBER"
    HEALTH_INSURANCE_NUMBER = "HEALTH_INSURANCE_NUMBER"

    # Generic
    IP_ADDRESS = "IP_ADDRESS"
    MAC_ADDRESS = "MAC_ADDRESS"
    UUID = "UUID"
    INTERNAL_HOSTNAME = "INTERNAL_HOSTNAME"


@dataclass
class DLPMatch:
    """Represents a single DLP match found in text."""
    violation_type: DLPViolationType
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str = ""
    masked_value: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "matched_text": self.matched_text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "context": self.context,
            "masked_value": self.masked_value
        }


@dataclass
class DLPResult:
    """Result of DLP inspection."""
    original_text: str
    processed_text: str
    violations: List[DLPMatch] = field(default_factory=list)
    action_taken: DLPAction = DLPAction.MASKING
    blocked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def violations_count(self) -> int:
        """Number of violations found."""
        return len(self.violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text_length": len(self.original_text),
            "processed_text": self.processed_text,
            "violations_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "action_taken": self.action_taken.value,
            "blocked": self.blocked,
            "metadata": self.metadata
        }


@dataclass
class DLPRule:
    """Configuration for a single DLP detection rule."""
    violation_type: DLPViolationType
    pattern: str
    confidence: float = 0.95
    validator: Optional[Callable[[str], bool]] = None
    context_window: int = 50
    action: DLPAction = DLPAction.MASKING
    enabled: bool = True


class DLPValidator:
    """Validators for reducing false positives."""

    @staticmethod
    def luhn_check(card_number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        digits = [int(d) for d in card_number if d.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
        checksum = 0
        is_even = False
        for d in reversed(digits):
            if is_even:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
            is_even = not is_even
        return checksum % 10 == 0

    @staticmethod
    def cpf_check(cpf: str) -> bool:
        """Validate Brazilian CPF."""
        digits = [int(d) for d in cpf if d.isdigit()]
        if len(digits) != 11 or len(set(digits)) == 1:
            return False
        # First check digit
        sum1 = sum(d * (10 - i) for i, d in enumerate(digits[:9]))
        check1 = (sum1 * 10) % 11
        if check1 == 10:
            check1 = 0
        if check1 != digits[9]:
            return False
        # Second check digit
        sum2 = sum(d * (11 - i) for i, d in enumerate(digits[:10]))
        check2 = (sum2 * 10) % 11
        if check2 == 10:
            check2 = 0
        return check2 == digits[10]

    @staticmethod
    def cnpj_check(cnpj: str) -> bool:
        """Validate Brazilian CNPJ."""
        digits = [int(d) for d in cnpj if d.isdigit()]
        if len(digits) != 14 or len(set(digits)) == 1:
            return False
        weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        sum1 = sum(d * w for d, w in zip(digits[:12], weights1))
        check1 = 11 - (sum1 % 11)
        if check1 >= 10:
            check1 = 0
        if check1 != digits[12]:
            return False
        sum2 = sum(d * w for d, w in zip(digits[:13], weights2))
        check2 = 11 - (sum2 % 11)
        if check2 >= 10:
            check2 = 0
        return check2 == digits[13]

    @staticmethod
    def jwt_check(token: str) -> bool:
        """Basic JWT structure validation (3 base64 parts)."""
        parts = token.split('.')
        if len(parts) != 3:
            return False
        try:
            import base64
            for part in parts[:2]:  # header and payload
                base64.urlsafe_b64decode(part + '=' * (-len(part) % 4))
            return True
        except Exception:
            return False

    @staticmethod
    def aws_key_check(key: str) -> bool:
        """Validate AWS access key format (AKIA... or ASIA...)."""
        # AWS access keys are 20 chars starting with AKIA, ASIA, AROA, AIDA
        valid_prefixes = ('AKIA', 'ASIA', 'AROA', 'AIDA')
        return key.startswith(valid_prefixes) and len(key) == 20

    @staticmethod
    def github_token_check(token: str) -> bool:
        """Validate GitHub token format."""
        return token.startswith(('ghp_', 'gho_', 'ghu_', 'ghs_', 'ghr_')) and len(token) >= 36

    @staticmethod
    def phone_br_check(phone: str) -> bool:
        """Validate Brazilian phone number format."""
        import re
        # Must have area code (2 digits) or country code +55
        # Formats: +55 (11) 99999-8888, (11) 99999-8888, 11 99999-8888, 99999-8888 (with context)
        # But we only accept if it has area code or country code
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 11:  # With area code (2) + 9 digits
            return digits[2] == '9'  # Mobile starts with 9
        elif len(digits) == 13 and digits.startswith('55'):  # +55 + area + number
            return digits[4] == '9'
        elif len(digits) == 10:  # Landline with area code
            return True
        return False

    @staticmethod
    def phone_us_check(phone: str) -> bool:
        """Validate US phone number format."""
        import re
        digits = re.sub(r'\D', '', phone)
        # US: 10 digits (with area code) or 11 digits (with country code 1)
        if len(digits) == 10:
            return True
        elif len(digits) == 11 and digits[0] == '1':
            return True
        return False

    @staticmethod
    def rg_check(rg: str) -> bool:
        """Validate Brazilian RG format."""
        import re
        # RG format: XX.XXX.XXX-X or XXXXXXX-X (9 digits + check digit)
        digits = re.sub(r'\D', '', rg)
        # Most RGs have 9 digits (8 + check) or 8 digits
        return len(digits) in (8, 9)


# Pre-compiled regex patterns for performance
DLP_PATTERNS: Dict[DLPViolationType, DLPRule] = {
    # ===== BRAZILIAN PII =====
    DLPViolationType.CPF: DLPRule(
        violation_type=DLPViolationType.CPF,
        pattern=r'\b\d{3}[.\s-]?\d{3}[.\s-]?\d{3}[-\s]?\d{2}\b',
        confidence=0.95,
        validator=DLPValidator.cpf_check,
        action=DLPAction.MASKING
    ),
    DLPViolationType.CNPJ: DLPRule(
        violation_type=DLPViolationType.CNPJ,
        pattern=r'\b\d{2}[.\s]?\d{3}[.\s]?\d{3}[/\s]?\d{4}[-\s]?\d{2}\b',
        confidence=0.95,
        validator=DLPValidator.cnpj_check,
        action=DLPAction.MASKING
    ),
    DLPViolationType.PHONE_BR: DLPRule(
        violation_type=DLPViolationType.PHONE_BR,
        # Require explicit formatting to avoid false positives in credit cards:
        # +55 (11) 99999-8888 or (11) 99999-8888 or 11 99999-8888 (with dash between last 8 digits)
        # NOT just spaces - must have dash or parentheses
        pattern=r'(?:\+55\s?)?(?:\(\d{2}\)\s?|\d{2}\s)(?:9\s?)?\d{4}-\d{4}\b',
        confidence=0.85,
        validator=DLPValidator.phone_br_check,
        action=DLPAction.MASKING
    ),
    DLPViolationType.EMAIL: DLPRule(
        violation_type=DLPViolationType.EMAIL,
        pattern=r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        confidence=0.98,
        action=DLPAction.MASKING
    ),
    DLPViolationType.CEP: DLPRule(
        violation_type=DLPViolationType.CEP,
        pattern=r'\b\d{5}[-\s]?\d{3}\b',
        confidence=0.85,
        action=DLPAction.MASKING
    ),
    DLPViolationType.RG: DLPRule(
        violation_type=DLPViolationType.RG,
        # More specific: XX.XXX.XXX-X or XX.XXX.XXX X or XXXXXXXX-X (8-9 digits with separators)
        pattern=r'\b\d{2}[.\s]\d{3}[.\s]\d{3}[-\s][0-9Xx]\b',
        confidence=0.85,
        validator=DLPValidator.rg_check,
        action=DLPAction.MASKING
    ),
    DLPViolationType.CNH: DLPRule(
        violation_type=DLPViolationType.CNH,
        pattern=r'\b\d{11}\b',
        confidence=0.70,
        action=DLPAction.MASKING
    ),
    DLPViolationType.PIS: DLPRule(
        violation_type=DLPViolationType.PIS,
        pattern=r'\b\d{3}[.\s]?\d{5}[.\s]?\d{2}[-\s]?\d\b',
        confidence=0.85,
        action=DLPAction.MASKING
    ),
    DLPViolationType.TITULO_ELEITOR: DLPRule(
        violation_type=DLPViolationType.TITULO_ELEITOR,
        pattern=r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b',
        confidence=0.80,
        action=DLPAction.MASKING
    ),

    # ===== INTERNATIONAL PII =====
    DLPViolationType.SSN_US: DLPRule(
        violation_type=DLPViolationType.SSN_US,
        pattern=r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        confidence=0.90,
        action=DLPAction.MASKING
    ),
    DLPViolationType.PASSPORT: DLPRule(
        violation_type=DLPViolationType.PASSPORT,
        pattern=r'\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b',
        confidence=0.75,
        action=DLPAction.MASKING
    ),
    DLPViolationType.PHONE_US: DLPRule(
        violation_type=DLPViolationType.PHONE_US,
        # Require area code with parentheses or explicit separator: +1 (555) 123-4567 or (555) 123-4567 or 555-123-4567
        pattern=r'(?:\+1\s?)?(?:\(\d{3}\)|\d{3}[-\s])\d{3}[-\s]\d{4}\b',
        confidence=0.80,
        validator=DLPValidator.phone_us_check,
        action=DLPAction.MASKING
    ),
    DLPViolationType.IBAN: DLPRule(
        violation_type=DLPViolationType.IBAN,
        pattern=r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',
        confidence=0.95,
        action=DLPAction.MASKING
    ),

    # ===== SECRETS & CREDENTIALS =====
    DLPViolationType.AWS_ACCESS_KEY: DLPRule(
        violation_type=DLPViolationType.AWS_ACCESS_KEY,
        pattern=r'\b(AKIA|ASIA|AROA|AIDA)[A-Z0-9]{16}\b',
        confidence=0.99,
        validator=DLPValidator.aws_key_check,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.AWS_SECRET_KEY: DLPRule(
        violation_type=DLPViolationType.AWS_SECRET_KEY,
        pattern=r'\b[A-Za-z0-9/+=]{40}\b',
        confidence=0.70,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.GCP_API_KEY: DLPRule(
        violation_type=DLPViolationType.GCP_API_KEY,
        pattern=r'\bAIza[0-9A-Za-z-_]{35}\b',
        confidence=0.99,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.GITHUB_TOKEN: DLPRule(
        violation_type=DLPViolationType.GITHUB_TOKEN,
        pattern=r'\bgh[pousr]_[A-Za-z0-9_]{36,255}\b',
        confidence=0.99,
        validator=DLPValidator.github_token_check,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.SLACK_TOKEN: DLPRule(
        violation_type=DLPViolationType.SLACK_TOKEN,
        pattern=r'\bxox[baprs]-[0-9]{12}-[0-9]{12}-[0-9]{12}-[a-z0-9]{32}\b',
        confidence=0.99,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.DISCORD_TOKEN: DLPRule(
        violation_type=DLPViolationType.DISCORD_TOKEN,
        pattern=r'\b[MN][A-Za-z\d]{23}\.[\w-]{6}\.[\w-]{27}\b',
        confidence=0.95,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.JWT_TOKEN: DLPRule(
        violation_type=DLPViolationType.JWT_TOKEN,
        pattern=r'\beyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b',
        confidence=0.90,
        validator=DLPValidator.jwt_check,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.DB_CONNECTION_STRING: DLPRule(
        violation_type=DLPViolationType.DB_CONNECTION_STRING,
        pattern=r'(?i)\b(?:postgresql|mysql|mongodb|redis|mssql|oracle)://[^\s]{10,200}\b',
        confidence=0.95,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.API_KEY_GENERIC: DLPRule(
        violation_type=DLPViolationType.API_KEY_GENERIC,
        pattern=r'(?i)(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token)["\s:=]+["\']?[A-Za-z0-9_\-]{20,}["\']?',
        confidence=0.85,
        action=DLPAction.MASKING
    ),
    DLPViolationType.PRIVATE_KEY: DLPRule(
        violation_type=DLPViolationType.PRIVATE_KEY,
        pattern=r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        confidence=0.99,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.SSH_PRIVATE_KEY: DLPRule(
        violation_type=DLPViolationType.SSH_PRIVATE_KEY,
        pattern=r'-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----',
        confidence=0.99,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.PASSWORD_IN_TEXT: DLPRule(
        violation_type=DLPViolationType.PASSWORD_IN_TEXT,
        pattern=r'(?i)(?:password|passwd|pwd|senha)["\s:=]+["\']?[^\s"\']{8,}["\']?',
        confidence=0.80,
        action=DLPAction.MASKING
    ),
    DLPViolationType.BEARER_TOKEN: DLPRule(
        violation_type=DLPViolationType.BEARER_TOKEN,
        pattern=r'(?i)bearer\s+[A-Za-z0-9_\-\.=]{20,}',
        confidence=0.90,
        action=DLPAction.MASKING
    ),

    # ===== FINANCIAL =====
    DLPViolationType.CREDIT_CARD: DLPRule(
        violation_type=DLPViolationType.CREDIT_CARD,
        pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        confidence=0.90,
        validator=DLPValidator.luhn_check,
        action=DLPAction.MASKING
    ),
    DLPViolationType.CREDIT_CARD_TRACK: DLPRule(
        violation_type=DLPViolationType.CREDIT_CARD_TRACK,
        pattern=r'%B\d{13,19}\^[\s\S]{1,50}\^\d{4}',
        confidence=0.95,
        action=DLPAction.BLOCK
    ),
    DLPViolationType.BANK_ACCOUNT_BR: DLPRule(
        violation_type=DLPViolationType.BANK_ACCOUNT_BR,
        pattern=r'\b\d{4}[-\s]\d{5,13}\b',  # Brazilian bank account: agência (4 digits) + separator + conta (5-13 digits)
        confidence=0.75,
        action=DLPAction.MASKING
    ),
    DLPViolationType.SWIFT_BIC: DLPRule(
        violation_type=DLPViolationType.SWIFT_BIC,
        pattern=r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b',
        confidence=0.85,
        action=DLPAction.MASKING
    ),
    DLPViolationType.CRYPTO_WALLET_BTC: DLPRule(
        violation_type=DLPViolationType.CRYPTO_WALLET_BTC,
        pattern=r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
        confidence=0.90,
        action=DLPAction.MASKING
    ),
    DLPViolationType.CRYPTO_WALLET_ETH: DLPRule(
        violation_type=DLPViolationType.CRYPTO_WALLET_ETH,
        pattern=r'\b0x[a-fA-F0-9]{40}\b',
        confidence=0.95,
        action=DLPAction.MASKING
    ),

    # ===== NETWORK/INFRASTRUCTURE =====
    DLPViolationType.IP_ADDRESS: DLPRule(
        violation_type=DLPViolationType.IP_ADDRESS,
        pattern=r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        confidence=0.90,
        action=DLPAction.MASKING
    ),
    DLPViolationType.MAC_ADDRESS: DLPRule(
        violation_type=DLPViolationType.MAC_ADDRESS,
        pattern=r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
        confidence=0.95,
        action=DLPAction.MASKING
    ),
    DLPViolationType.UUID: DLPRule(
        violation_type=DLPViolationType.UUID,
        pattern=r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
        confidence=0.98,
        action=DLPAction.MASKING
    ),
    DLPViolationType.INTERNAL_HOSTNAME: DLPRule(
        violation_type=DLPViolationType.INTERNAL_HOSTNAME,
        pattern=r'\b(?:[a-z0-9-]+\.)*(?:internal|corp|local|intranet|dev|staging|prod|prd|homolog)\b',
        confidence=0.75,
        action=DLPAction.MASKING
    ),
}


class DLPEngine:
    """
    Main DLP Engine for detecting and mitigating sensitive data exposure.

    Supports configurable policies, custom rules, and integration with
    deception orchestrator for honeytoken injection.
    """

    def __init__(
        self,
        default_action: DLPAction = DLPAction.MASKING,
        custom_rules: Optional[List[DLPRule]] = None,
        enabled_types: Optional[List[DLPViolationType]] = None,
        honeytoken_callback: Optional[Callable[[DLPViolationType, str], str]] = None,
        log_violations: bool = True
    ):
        """
        Initialize DLP Engine.

        Args:
            default_action: Default mitigation action for violations
            custom_rules: Additional custom DLPRule instances
            enabled_types: Subset of violation types to enable (None = all)
            honeytoken_callback: Callback for HONEYTOKEN_INJECTION action
                Signature: (violation_type, original_value) -> honeytoken_value
            log_violations: Whether to log violations to logger
        """
        self.default_action = default_action
        self.honeytoken_callback = honeytoken_callback
        self.log_violations = log_violations
        self.enabled_types = set(enabled_types) if enabled_types else set(DLPViolationType)

        # Compile all rules - create copies to avoid modifying global DLP_PATTERNS
        self.rules: Dict[DLPViolationType, DLPRule] = {}
        for vtype, rule in DLP_PATTERNS.items():
            if vtype in self.enabled_types:
                # Create a copy of the rule to avoid modifying the global pattern
                rule_copy = DLPRule(
                    violation_type=rule.violation_type,
                    pattern=rule.pattern,
                    confidence=rule.confidence,
                    validator=rule.validator,
                    context_window=rule.context_window,
                    action=rule.action,
                    enabled=rule.enabled
                )
                rule_copy.compiled_pattern = re.compile(rule_copy.pattern, re.IGNORECASE | re.MULTILINE)
                self.rules[vtype] = rule_copy

        # Add custom rules
        if custom_rules:
            for rule in custom_rules:
                if rule.violation_type in self.enabled_types:
                    rule_copy = DLPRule(
                        violation_type=rule.violation_type,
                        pattern=rule.pattern,
                        confidence=rule.confidence,
                        validator=rule.validator,
                        context_window=rule.context_window,
                        action=rule.action,
                        enabled=rule.enabled
                    )
                    rule_copy.compiled_pattern = re.compile(rule_copy.pattern, re.IGNORECASE | re.MULTILINE)
                    self.rules[rule.violation_type] = rule_copy

        logger.info(f"DLP Engine initialized with {len(self.rules)} rules, default_action={default_action.value}")

    def inspect(
        self,
        text: str,
        direction: str = "outbound",
        context: Optional[Dict[str, Any]] = None
    ) -> DLPResult:
        """
        Inspect text for sensitive data violations.

        Args:
            text: Text to inspect
            direction: "inbound" (user -> LLM) or "outbound" (LLM -> user)
            context: Optional context metadata (session_id, user_id, etc.)

        Returns:
            DLPResult with processed text and violations found
        """
        if not text or not isinstance(text, str):
            return DLPResult(original_text=text or "", processed_text=text or "")

        violations = []
        processed_text = text

        # Sort matches by position (descending) for safe replacement
        all_matches = []

        for vtype, rule in self.rules.items():
            if not rule.enabled:
                continue

            for match in rule.compiled_pattern.finditer(text):
                matched_text = match.group()
                start, end = match.span()

                # Run validator if provided
                if rule.validator and not rule.validator(matched_text):
                    continue

                # Extract context
                ctx_start = max(0, start - rule.context_window)
                ctx_end = min(len(text), end + rule.context_window)
                context_text = text[ctx_start:ctx_end]

                dlp_match = DLPMatch(
                    violation_type=vtype,
                    matched_text=matched_text,
                    start_pos=start,
                    end_pos=end,
                    confidence=rule.confidence,
                    context=context_text
                )
                all_matches.append((start, end, dlp_match, rule))

        # Sort by start position descending for safe string replacement
        all_matches.sort(key=lambda x: x[0], reverse=True)

        # Apply mitigations
        for start, end, match, rule in all_matches:
            action = rule.action if rule.action != DLPAction.MASKING else self.default_action

            if action == DLPAction.HONEYTOKEN_INJECTION and self.honeytoken_callback:
                match.masked_value = self.honeytoken_callback(match.violation_type, match.matched_text)
            elif action in (DLPAction.MASKING, DLPAction.REDACTION):
                match.masked_value = self._generate_mask(match.violation_type, action)
            elif action == DLPAction.BLOCK:
                match.masked_value = f"[BLOCKED:{match.violation_type.value}]"

            # Apply replacement
            processed_text = processed_text[:start] + match.masked_value + processed_text[end:]
            violations.append(match)

            if self.log_violations:
                logger.warning(
                    f"DLP Violation: type={match.violation_type.value} "
                    f"direction={direction} confidence={match.confidence:.2f} "
                    f"action={action.value} context={context}"
                )

        # Check if any BLOCK action was triggered
        blocked = any(
            self.rules[v.violation_type].action == DLPAction.BLOCK or
            (self.rules[v.violation_type].action == DLPAction.MASKING and self.default_action == DLPAction.BLOCK)
            for v in violations
        )

        result = DLPResult(
            original_text=text,
            processed_text=processed_text,
            violations=violations,
            action_taken=self.default_action,
            blocked=blocked,
            metadata={
                "direction": direction,
                "context": context or {},
                "rules_evaluated": len(self.rules)
            }
        )

        return result

    def _generate_mask(self, violation_type: DLPViolationType, action: DLPAction) -> str:
        """Generate masked placeholder for violation type."""
        if action == DLPAction.REDACTION:
            return ""
        type_name = violation_type.value
        return f"[{type_name}_REDACTED]"

    def inspect_batch(
        self,
        texts: List[str],
        direction: str = "outbound",
        context: Optional[Dict[str, Any]] = None
    ) -> List[DLPResult]:
        """Inspect multiple texts in batch."""
        return [self.inspect(text, direction, context) for text in texts]

    def add_custom_rule(self, rule: DLPRule) -> None:
        """Add a custom rule at runtime."""
        if rule.violation_type in self.enabled_types:
            rule.compiled_pattern = re.compile(rule.pattern, re.IGNORECASE | re.MULTILINE)
            self.rules[rule.violation_type] = rule
            logger.info(f"Added custom DLP rule: {rule.violation_type.value}")

    def remove_rule(self, violation_type: DLPViolationType) -> None:
        """Disable a rule at runtime."""
        if violation_type in self.rules:
            self.rules[violation_type].enabled = False
            logger.info(f"Disabled DLP rule: {violation_type.value}")

    def get_rule_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded rules."""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "by_action": {
                action.value: sum(1 for r in self.rules.values() if r.action == action)
                for action in DLPAction
            },
            "violation_types": [v.value for v in self.rules.keys()]
        }


# ===== INTEGRATION HELPERS =====

class DLPMiddleware:
    """
    Middleware for integrating DLP into request/response pipelines.
    """

    def __init__(self, engine: DLPEngine):
        self.engine = engine

    async def inspect_request(self, prompt: str, session_id: str = "") -> DLPResult:
        """Inspect inbound request (user prompt to LLM)."""
        return self.engine.inspect(
            prompt,
            direction="inbound",
            context={"session_id": session_id, "stage": "request"}
        )

    async def inspect_response(
        self,
        response: str,
        session_id: str = "",
        prompt: str = ""
    ) -> DLPResult:
        """Inspect outbound response (LLM output to user)."""
        return self.engine.inspect(
            response,
            direction="outbound",
            context={"session_id": session_id, "stage": "response", "prompt_hash": hash(prompt)}
        )

    async def process_chat_completion(
        self,
        messages: List[Dict[str, str]],
        session_id: str = ""
    ) -> tuple[List[Dict[str, str]], List[DLPResult]]:
        """
        Process full chat completion pipeline.

        Returns:
            Tuple of (processed_messages, inspection_results)
        """
        results = []
        processed_messages = []

        # Inspect user messages (inbound)
        for msg in messages:
            if msg.get("role") == "user":
                result = await self.inspect_request(msg["content"], session_id)
                results.append(result)
                if result.blocked:
                    # Return blocked response
                    return [{
                        "role": "assistant",
                        "content": "Sua mensagem foi bloqueada por violar políticas de segurança de dados (DLP)."
                    }], results
                processed_messages.append({
                    **msg,
                    "content": result.processed_text
                })
            else:
                processed_messages.append(msg)

        return processed_messages, results


# ===== HONEYTOKEN GENERATOR =====

class HoneytokenGenerator:
    """
    Generates realistic but fake honeytokens for deception.
    Integrates with deception-orchestrator for tracking.
    """

    # Fake but realistic generators
    @staticmethod
    def fake_cpf() -> str:
        import random
        # Generate valid CPF
        digits = [random.randint(0, 9) for _ in range(9)]
        sum1 = sum(d * (10 - i) for i, d in enumerate(digits))
        check1 = (sum1 * 10) % 11
        check1 = 0 if check1 == 10 else check1
        digits.append(check1)
        sum2 = sum(d * (11 - i) for i, d in enumerate(digits))
        check2 = (sum2 * 10) % 11
        check2 = 0 if check2 == 10 else check2
        digits.append(check2)
        return f"{digits[0]}{digits[1]}{digits[2]}.{digits[3]}{digits[4]}{digits[5]}.{digits[6]}{digits[7]}{digits[8]}-{digits[9]}{digits[10]}"

    @staticmethod
    def fake_cnpj() -> str:
        import random
        digits = [random.randint(0, 9) for _ in range(12)]
        weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        sum1 = sum(d * w for d, w in zip(digits, weights1))
        check1 = 11 - (sum1 % 11)
        check1 = 0 if check1 >= 10 else check1
        digits.append(check1)
        sum2 = sum(d * w for d, w in zip(digits, weights2))
        check2 = 11 - (sum2 % 11)
        check2 = 0 if check2 >= 10 else check2
        digits.append(check2)
        return f"{digits[0]}{digits[1]}.{digits[2]}{digits[3]}{digits[4]}.{digits[5]}{digits[6]}{digits[7]}/{digits[8]}{digits[9]}{digits[10]}{digits[11]}-{digits[12]}{digits[13]}"

    @staticmethod
    def fake_credit_card() -> str:
        import random
        # Generate valid Luhn credit card (Visa prefix 4)
        # Start with 15 digits (prefix + 14 random), compute 16th as check digit
        digits = [4] + [random.randint(0, 9) for _ in range(14)]

        # Calculate check digit using Luhn algorithm
        # Process digits from right to left (excluding check digit position)
        checksum = 0
        is_second = True  # Start from second-to-last digit
        for d in reversed(digits):
            if is_second:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
            is_second = not is_second
        check_digit = (10 - (checksum % 10)) % 10
        digits.append(check_digit)
        return f"{digits[0]}{digits[1]}{digits[2]}{digits[3]} {digits[4]}{digits[5]}{digits[6]}{digits[7]} {digits[8]}{digits[9]}{digits[10]}{digits[11]} {digits[12]}{digits[13]}{digits[14]}{digits[15]}"

    @staticmethod
    def fake_api_key(prefix: str = "sk") -> str:
        import secrets
        return f"{prefix}-{secrets.token_urlsafe(32)}"

    @staticmethod
    def fake_jwt() -> str:
        import base64, json, secrets
        header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip('=')
        payload = base64.urlsafe_b64encode(json.dumps({
            "sub": "honeytoken-user",
            "iat": 1699999999,
            "exp": 1799999999,
            "honeytoken": True
        }).encode()).decode().rstrip('=')
        signature = secrets.token_urlsafe(32)
        return f"{header}.{payload}.{signature}"

    @staticmethod
    def fake_email() -> str:
        import random, string
        domains = ["example.com", "test.org", "fake.local", "decoy.net"]
        name = ''.join(random.choices(string.ascii_lowercase, k=8))
        return f"{name}@{random.choice(domains)}"

    @classmethod
    def generate_for_type(cls, violation_type: DLPViolationType) -> str:
        """Generate appropriate honeytoken for violation type."""
        generators = {
            DLPViolationType.CPF: cls.fake_cpf,
            DLPViolationType.CNPJ: cls.fake_cnpj,
            DLPViolationType.CREDIT_CARD: cls.fake_credit_card,
            DLPViolationType.API_KEY_GENERIC: cls.fake_api_key,
            DLPViolationType.JWT_TOKEN: cls.fake_jwt,
            DLPViolationType.EMAIL: cls.fake_email,
            DLPViolationType.AWS_ACCESS_KEY: lambda: f"AKIA{secrets.token_urlsafe(16)}",
            DLPViolationType.GITHUB_TOKEN: lambda: f"ghp_{secrets.token_urlsafe(36)}",
        }
        import secrets
        generator = generators.get(violation_type, lambda: f"[HONEYTOKEN:{violation_type.value}]")
        return generator()


# ===== FACTORY FUNCTIONS =====

def create_dlp_engine(
    profile: str = "balanced",
    honeytoken_enabled: bool = False,
    custom_rules: Optional[List[DLPRule]] = None
) -> DLPEngine:
    """
    Factory function to create DLP engine with predefined profiles.

    Profiles:
    - "strict": BLOCK on secrets, MASKING on PII
    - "balanced": MASKING on all, BLOCK only on critical secrets
    - "monitor": MASKING on all, no blocking (audit mode)
    - "lgpd": Brazilian LGPD compliance focus
    - "pci-dss": PCI-DSS compliance focus
    """
    if profile == "strict":
        default_action = DLPAction.BLOCK
        enabled_types = None  # All types
    elif profile == "balanced":
        default_action = DLPAction.MASKING
        enabled_types = None
    elif profile == "monitor":
        default_action = DLPAction.MASKING
        enabled_types = None
    elif profile == "lgpd":
        default_action = DLPAction.MASKING
        enabled_types = [
            DLPViolationType.CPF, DLPViolationType.CNPJ, DLPViolationType.PHONE_BR,
            DLPViolationType.EMAIL, DLPViolationType.CEP, DLPViolationType.RG,
            DLPViolationType.CNH, DLPViolationType.PIS,
            DLPViolationType.CREDIT_CARD, DLPViolationType.BANK_ACCOUNT_BR,
            DLPViolationType.API_KEY_GENERIC, DLPViolationType.JWT_TOKEN,
            DLPViolationType.DB_CONNECTION_STRING, DLPViolationType.PASSWORD_IN_TEXT
        ]
    elif profile == "pci-dss":
        default_action = DLPAction.BLOCK
        enabled_types = [
            DLPViolationType.CREDIT_CARD, DLPViolationType.CREDIT_CARD_TRACK,
            DLPViolationType.BANK_ACCOUNT_BR, DLPViolationType.BANK_ACCOUNT_US,
            DLPViolationType.SWIFT_BIC, DLPViolationType.API_KEY_GENERIC,
            DLPViolationType.DB_CONNECTION_STRING, DLPViolationType.PRIVATE_KEY
        ]
    else:
        default_action = DLPAction.MASKING
        enabled_types = None

    honeytoken_cb = None
    if honeytoken_enabled:
        honeytoken_cb = HoneytokenGenerator.generate_for_type

    return DLPEngine(
        default_action=default_action,
        custom_rules=custom_rules,
        enabled_types=enabled_types,
        honeytoken_callback=honeytoken_cb,
        log_violations=True
    )


# ===== EXPORTS =====

__all__ = [
    # Enums
    "DLPAction",
    "DLPViolationType",
    # Data classes
    "DLPMatch",
    "DLPResult",
    "DLPRule",
    # Validators
    "DLPValidator",
    # Core engine
    "DLPEngine",
    "DLP_PATTERNS",
    # Middleware
    "DLPMiddleware",
    # Honeytoken
    "HoneytokenGenerator",
    # Factory
    "create_dlp_engine",
]