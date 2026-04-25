"""Core audit engine: feeds claim + PM-JAY rules to LLM and returns structured decision."""

import json
import re
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, FRAUD_THRESHOLD

AUDIT_PROMPT = """\
You are a senior medical claim auditor for the Ayushman Bharat PM-JAY scheme. \
Your job is to detect fraud, upcoding, and non-compliance.

=== PATIENT CLAIM ===
{claim_text}

=== RELEVANT PM-JAY GUIDELINES (retrieved) ===
{rules_text}

=== AUDIT TASK ===
Carefully compare the claim against the PM-JAY guidelines. Check for:
1. Length of stay exceeding the PM-JAY maximum for the procedure.
2. Claimed amount exceeding the PM-JAY package rate.
3. Procedures that do not match the stated diagnosis.
4. Phantom procedures (billed but clinically unjustified).
5. Duplicate billing or impossible clinical combinations.

Respond ONLY with a valid JSON object — no extra text:
{{
  "status": "APPROVED" or "FLAGGED",
  "risk_score": <float 0.0–1.0 where 1.0 = certain fraud>,
  "reason": "<one concise sentence>",
  "cited_rule": "<exact PM-JAY rule or package that applies>",
  "violations": ["<specific issue 1>", "<specific issue 2>"],
  "recommended_action": "<what the auditor should do next>"
}}
"""


class AuditEngine:
    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model

    def audit_claim(self, claim_text: str, relevant_rules: list[str]) -> dict:
        rules_text = "\n\n---\n\n".join(relevant_rules)
        prompt = AUDIT_PROMPT.format(claim_text=claim_text, rules_text=rules_text)

        start = time.time()
        raw_response = self._call_llm(prompt)
        elapsed = round(time.time() - start, 2)

        result = self._parse_response(raw_response)
        result["latency_seconds"] = elapsed
        result["llm_model"] = self.model
        result["status"] = (
            "FLAGGED"
            if result.get("risk_score", 0) >= FRAUD_THRESHOLD
            else result.get("status", "APPROVED")
        )
        return result

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "sarvam":
            return self._call_sarvam(prompt)
        elif self.provider == "databricks":
            return self._call_databricks(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        return self._call_mock(prompt)

    def _call_sarvam(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": "sarvam-m",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_databricks(self, prompt: str) -> str:
        import requests
        from config import DATABRICKS_HOST, DATABRICKS_TOKEN
        endpoint = f"{DATABRICKS_HOST}/serving-endpoints/{self.model}/invocations"
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        msg = client.messages.create(
            model=self.model or "claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _call_mock(self, prompt: str) -> str:
        """Rule-based fallback when no LLM is configured — for offline demos."""
        claim_lower = prompt.lower()
        violations = []
        risk = 0.1

        if re.search(r"(\d+)\s*day", claim_lower):
            days_match = re.search(r"(\d+)\s*day", claim_lower)
            days = int(days_match.group(1)) if days_match else 0
            if days > 5 and ("cataract" in claim_lower or "phaco" in claim_lower):
                violations.append(f"Length of stay {days} days exceeds PM-JAY max (1 day) for cataract surgery")
                risk = 0.92

        if re.search(r"₹\s*([\d,]+)", claim_lower):
            amounts = [int(m.replace(",", "")) for m in re.findall(r"[\d,]+", re.search(r"₹\s*([\d,]+)", claim_lower).group(0))]
            if amounts and amounts[0] > 10000 and "cataract" in claim_lower:
                violations.append(f"Claimed ₹{amounts[0]:,} exceeds PM-JAY package H/03 rate of ₹10,000")
                risk = max(risk, 0.85)

        if "ghost" in claim_lower or "phantom" in claim_lower:
            violations.append("Suspected ghost patient — no verifiable admission records")
            risk = 0.97

        status = "FLAGGED" if risk >= FRAUD_THRESHOLD else "APPROVED"
        return json.dumps({
            "status": status,
            "risk_score": risk,
            "reason": violations[0] if violations else "Claim complies with PM-JAY guidelines",
            "cited_rule": "PM-JAY HBP 2022 — Package H/03 (Cataract)" if "cataract" in claim_lower else "PM-JAY STG v2.0",
            "violations": violations,
            "recommended_action": "Escalate to senior auditor" if status == "FLAGGED" else "Approve for reimbursement",
        })

    @staticmethod
    def _parse_response(raw: str) -> dict:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {
            "status": "ERROR",
            "risk_score": 0.0,
            "reason": "LLM returned unparseable response",
            "cited_rule": "",
            "violations": [],
            "recommended_action": "Manual review required",
            "raw_response": raw[:500],
        }
