"""
Quick offline demo — no Databricks needed.
Builds FAISS index from sample rules, audits all synthetic claims, prints results.

Usage:  python run_demo.py
"""

import sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.translator import translate_to_english
from utils.vector_store import build_index, retrieve_relevant_rules
from utils.auditor import AuditEngine

RULES_FILE   = Path("data/pmjay_rules/hbp_rules_sample.txt")
CLAIMS_DIR   = Path("data/synthetic_claims")
INDEX_PATH   = Path("/tmp/pmjay_demo_index")

RESET        = "\033[0m"
GREEN        = "\033[92m"
RED          = "\033[91m"
YELLOW       = "\033[93m"
CYAN         = "\033[96m"
BOLD         = "\033[1m"


def banner(text, color=CYAN):
    print(f"\n{color}{BOLD}{'='*60}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{'='*60}{RESET}")


def main():
    banner("AyushAudit AI — PM-JAY Demo (Offline Mode)")

    # 1. Build vector index
    print(f"\n{YELLOW}► Building PM-JAY rules vector index…{RESET}")
    rules_text = RULES_FILE.read_text(encoding="utf-8")
    build_index(rules_text, str(INDEX_PATH))
    print(f"  Index built. ({len(rules_text):,} chars → FAISS)")

    # 2. Audit each synthetic claim
    engine = AuditEngine(provider="mock")
    claim_files = sorted(CLAIMS_DIR.glob("*.txt"))

    results = []
    for claim_file in claim_files:
        print(f"\n{YELLOW}► Processing: {claim_file.name}{RESET}")
        raw_text = claim_file.read_text(encoding="utf-8")

        # Translate
        tr = translate_to_english(raw_text)
        if tr["translation_needed"]:
            print(f"  Language: {tr['source_language_name']} → translated via {tr['method']}")
        else:
            print(f"  Language: English (no translation needed)")

        # Retrieve rules
        rules = retrieve_relevant_rules(tr["translated_text"], k=5)

        # Audit
        t0 = time.time()
        audit = engine.audit_claim(tr["translated_text"], rules)
        elapsed = round(time.time() - t0, 3)

        status   = audit["status"]
        risk     = audit.get("risk_score", 0)
        reason   = audit.get("reason", "")
        rule_ref = audit.get("cited_rule", "")

        color = GREEN if status == "APPROVED" else RED
        print(f"  {color}{BOLD}[{status}]{RESET}  Risk: {risk:.0%}  ({elapsed}s)")
        print(f"  Reason: {reason}")
        print(f"  Rule:   {rule_ref}")
        if audit.get("violations"):
            for v in audit["violations"]:
                print(f"  {RED}⛔ {v}{RESET}")

        results.append({"file": claim_file.name, "status": status, "risk": risk})

    # 3. Summary
    banner("BATCH SUMMARY", CYAN)
    flagged = [r for r in results if r["status"] == "FLAGGED"]
    approved = [r for r in results if r["status"] == "APPROVED"]

    print(f"\n  Total Claims : {len(results)}")
    print(f"  {GREEN}Approved     : {len(approved)}{RESET}")
    print(f"  {RED}Flagged      : {len(flagged)}{RESET}")
    print(f"  Fraud Rate   : {len(flagged)/len(results):.0%}" if results else "")

    print(f"\n{YELLOW}► To launch the full Streamlit UI:{RESET}")
    print(f"  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
