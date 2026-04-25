"""
Generate additional synthetic PM-JAY medical claims using Claude/OpenAI.
Run: python data/generate_synthetic_data.py --count 20 --output data/synthetic_claims/
"""

import argparse
import json
import os
import random
from pathlib import Path

SCENARIOS = [
    {"type": "compliant", "procedure": "cataract", "lang": "en"},
    {"type": "fraud_overstay", "procedure": "cataract", "lang": "hi"},
    {"type": "fraud_upcoding", "procedure": "appendectomy", "lang": "en"},
    {"type": "compliant", "procedure": "knee_replacement", "lang": "ta"},
    {"type": "fraud_phantom", "procedure": "bypass", "lang": "en"},
    {"type": "compliant", "procedure": "normal_delivery", "lang": "en"},
    {"type": "fraud_overstay", "procedure": "laparoscopic_chole", "lang": "en"},
]

GENERATION_PROMPT = """\
Generate a realistic synthetic PM-JAY (Ayushman Bharat) hospital discharge summary.

Scenario type: {scenario_type}
Procedure: {procedure}
Language: {language}

Rules for the scenario type:
- "compliant": All details (stay duration, claimed amount, diagnosis) exactly match PM-JAY HBP guidelines.
- "fraud_overstay": The stay duration significantly exceeds the PM-JAY maximum for this procedure, but no clinical justification is given.
- "fraud_upcoding": The hospital bills for a more expensive PM-JAY package than what was actually performed (the OT records contradict the bill).
- "fraud_phantom": The procedures listed on the bill were never actually performed; investigation records are missing or inconsistent.

Include: Patient name, age, diagnosis, ICD-10 code, procedure done, length of stay, claimed amount, PM-JAY package code, attending surgeon, and clinical notes.

If language is "hi", write the entire document in Hindi Devanagari script.
If language is "ta", write the entire document in Tamil script.
Otherwise write in English.

Make the document realistic, 200-300 words. Do NOT include any real patient data.
"""


def generate_with_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def generate_with_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", default="data/synthetic_claims")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_fn = generate_with_anthropic if args.provider == "anthropic" else generate_with_openai

    for i in range(args.count):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        prompt = GENERATION_PROMPT.format(
            scenario_type=scenario["type"],
            procedure=scenario["procedure"],
            language=scenario["lang"],
        )
        print(f"[{i+1}/{args.count}] Generating {scenario['type']} / {scenario['procedure']} ({scenario['lang']})…")
        try:
            text = generate_fn(prompt)
            fname = out_dir / f"gen_{i+1:03d}_{scenario['lang']}_{scenario['type']}.txt"
            fname.write_text(text, encoding="utf-8")
            print(f"  Saved: {fname.name}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
