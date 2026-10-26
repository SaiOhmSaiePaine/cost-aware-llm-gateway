# test_openai_sdk.py
"""End-to-end verification using the official OpenAI Python SDK against the local gateway."""
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("GATEWAY_BASE_URL", "http://localhost:8000/v1"),
    api_key="not-needed-local-gateway",
)

prompts = [
    "Translate 'hello friend' into Spanish.",
    "Can you translate hello friend to Spanish?",  # Should hit ONNX + Redis Cache!
]

for p in prompts:
    print(f"\nSending: {p}")
    raw = client.chat.completions.with_raw_response.create(
        model="auto-router",
        messages=[{"role": "user", "content": p}],
    )
    response = raw.parse()
    print("AI Answer:", (response.choices[0].message.content or "").strip())
    print("Gateway Telemetry:", response.model_extra.get("gateway_metadata"))
    print(
        "Headers:",
        {k: raw.headers.get(k) for k in ("x-gateway-route", "x-gateway-latency-ms", "x-gateway-cache-hit")},
    )
