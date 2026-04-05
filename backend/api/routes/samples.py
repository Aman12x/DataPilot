"""
backend/api/routes/samples.py

GET /samples          → list of available sample datasets
GET /samples/{name}   → stream the CSV file (no auth — demo data)
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["samples"])

_SAMPLES_DIR = os.getenv("SAMPLES_DIR", "data/samples")

# Allowlist — only these files can be served (prevents path traversal)
SAMPLES = [
    {
        "name":           "ecommerce_ab_test.csv",
        "label":          "E-commerce A/B Test",
        "domain":         "E-commerce",
        "icon":           "🛒",
        "mode":           "ab_test",
        "suggested_task": "Did the new checkout flow increase revenue and orders? Identify which customer segments and devices benefited most.",
    },
    {
        "name":           "clinical_trial.csv",
        "label":          "Clinical Trial",
        "domain":         "Healthcare",
        "icon":           "🏥",
        "mode":           "ab_test",
        "suggested_task": "Did Drug A improve recovery scores compared to the control group? Check for side effects and subgroup differences by age and severity.",
    },
    {
        "name":           "saas_churn_analysis.csv",
        "label":          "SaaS Churn Analysis",
        "domain":         "SaaS",
        "icon":           "📊",
        "mode":           "general",
        "suggested_task": "What factors most strongly predict customer churn? Which industries and plan types have the highest churn rates?",
    },
    {
        "name":           "logistics_ops.csv",
        "label":          "Logistics Operations",
        "domain":         "Logistics",
        "icon":           "🚚",
        "mode":           "general",
        "suggested_task": "Why are deliveries being delayed? Which carriers and routes have the worst on-time performance?",
    },
    {
        "name":           "media_ctr_experiment.csv",
        "label":          "Media CTR Experiment",
        "domain":         "Media",
        "icon":           "📱",
        "mode":           "ab_test",
        "suggested_task": "Did the new content recommendation algorithm improve click-through rate and watch time? Break down results by platform, content type, and age group.",
    },
    {
        "name":           "customer_transactions_10k.csv",
        "label":          "Retail Transactions",
        "domain":         "Retail",
        "icon":           "🏪",
        "mode":           "general",
        "suggested_task": "Which product categories and customer segments drive the most revenue? What patterns exist in fraud, returns, and churn?",
    },
]

_ALLOWED = {s["name"] for s in SAMPLES}


@router.get("/samples")
def list_samples() -> list[dict]:
    return SAMPLES


@router.get("/samples/{name}")
def get_sample(name: str):
    if name not in _ALLOWED:
        raise HTTPException(status_code=404, detail="Sample not found")
    path = Path(_SAMPLES_DIR) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Sample file not found on server")
    return FileResponse(path, media_type="text/csv", filename=name)
