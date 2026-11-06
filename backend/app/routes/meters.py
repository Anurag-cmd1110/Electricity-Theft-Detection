from datetime import datetime, timezone
from pathlib import Path
import math
import random
import re
import sys
from typing import List
import io
from bson import ObjectId

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd
from pypdf import PdfReader

from app.database import meter_readings, investigations, meters_collection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict import predict_meter

router = APIRouter(prefix="/meters", tags=["Meters"])

DEFAULT_FEEDER_ZONES = [
    "Sector 4 Feeder (Substation A)",
    "Morabadi Feeder (Substation B)",
    "Kanke Road Feeder (Substation C)",
    "Harmu Feeder (Substation D)",
    "Doranda Substation Feeder",
]


# --------------------------------------------------
# Schemas
# --------------------------------------------------

class RegisterMeterRequest(BaseModel):
    meter_id: str = Field(min_length=1, max_length=100)
    consumer_name: str = Field(min_length=1, max_length=150)
    sanctioned_load_kw: float = Field(gt=0)
    meter_type: str = Field(default="Single Phase")
    area_zone: str = Field(default="", max_length=100)


class UpdateConsumerNameRequest(BaseModel):
    consumer_name: str = Field(min_length=1, max_length=150)


class MLPredictionRequest(BaseModel):
    meter_id: str = Field(default="MTR-001", min_length=1, max_length=100)
    previous_month_kwh: float = Field(ge=0, allow_inf_nan=False)
    current_month_kwh: float = Field(ge=0, allow_inf_nan=False)
    avg_daily_kwh: float = Field(ge=0, allow_inf_nan=False)
    usage_change_pct: float = Field(allow_inf_nan=False)
    night_usage_ratio: float = Field(ge=0, le=1.5, allow_inf_nan=False)
    weekend_usage_ratio: float = Field(ge=0, le=2.0, allow_inf_nan=False)
    zero_reading_days: int = Field(ge=0)


class UpdateInvestigationStatusRequest(BaseModel):
    status: str = Field(pattern="^(pending|under_investigation|resolved_theft|false_positive)$")
    notes: str = Field(default="")


# --------------------------------------------------
# Meter Registry Endpoints
# --------------------------------------------------

@router.get("/registry")
def list_meters(
    search: str = Query(default=""),
    status: str = Query(default="all"),
):
    query = {}
    if search.strip():
        query["$or"] = [
            {"meter_id": {"$regex": search.strip(), "$options": "i"}},
            {"consumer_name": {"$regex": search.strip(), "$options": "i"}},
            {"area_zone": {"$regex": search.strip(), "$options": "i"}},
        ]
    if status != "all":
        query["status"] = status

    try:
        registered = list(meters_collection.find(query).sort("created_at", -1))
    except Exception as error:
        raise HTTPException(status_code=503, detail="Unable to fetch meters from MongoDB.") from error

    for m in registered:
        m["id"] = str(m.pop("_id"))
        if isinstance(m.get("created_at"), datetime):
            m["created_at"] = m["created_at"].isoformat()

    return {"count": len(registered), "meters": registered}


@router.patch("/registry/{meter_id}/consumer")
def update_consumer_name(meter_id: str, body: UpdateConsumerNameRequest):
    new_name = body.consumer_name.strip()
    result = meters_collection.update_one(
        {"meter_id": meter_id},
        {"$set": {"consumer_name": new_name, "updated_at": datetime.now(timezone.utc)}},
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Meter '{meter_id}' not found in registry.")

    return {
        "message": f"Consumer name for meter '{meter_id}' updated successfully.",
        "meter_id": meter_id,
        "consumer_name": new_name,
    }


@router.post("/registry/sync-from-history")
def sync_meters_from_history():
    """Populate and redistribute all meters found in reading history across multiple feeders."""
    unique_meter_ids = meter_readings.distinct("meter_id")
    synced_count = 0
    now = datetime.now(timezone.utc)

    for idx, mid in enumerate(unique_meter_ids):
        if not mid or mid == "Unknown":
            continue

        assigned_zone = DEFAULT_FEEDER_ZONES[idx % len(DEFAULT_FEEDER_ZONES)]
        existing = meters_collection.find_one({"meter_id": mid})

        if not existing:
            meters_collection.insert_one({
                "meter_id": mid,
                "consumer_name": f"Consumer ({mid})",
                "sanctioned_load_kw": 5.0,
                "meter_type": "Three Phase Smart",
                "area_zone": assigned_zone,
                "status": "Active",
                "created_at": now,
            })
            synced_count += 1
        elif (
            not existing.get("area_zone")
            or existing.get("area_zone") == "General Feeder"
            or existing.get("area_zone") == "General Substation"
        ):
            meters_collection.update_one(
                {"meter_id": mid},
                {"$set": {"area_zone": assigned_zone, "updated_at": now}}
            )
            synced_count += 1

    return {
        "message": f"Successfully synced and distributed {synced_count} meters across substations.",
        "synced": synced_count,
    }


@router.post("/registry")
def register_meter(meter_data: RegisterMeterRequest):
    meter_id = meter_data.meter_id.strip()

    existing = meters_collection.find_one({"meter_id": meter_id})
    if existing:
        raise HTTPException(status_code=400, detail=f"Meter '{meter_id}' is already registered.")

    assigned_zone = meter_data.area_zone.strip() if meter_data.area_zone.strip() else random.choice(DEFAULT_FEEDER_ZONES)
    now = datetime.now(timezone.utc)
    doc = {
        "meter_id": meter_id,
        "consumer_name": meter_data.consumer_name.strip(),
        "sanctioned_load_kw": meter_data.sanctioned_load_kw,
        "meter_type": meter_data.meter_type,
        "area_zone": assigned_zone,
        "status": "Active",
        "created_at": now,
    }

    try:
        saved = meters_collection.insert_one(doc)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Could not register meter in MongoDB.") from error

    return {
        "id": str(saved.inserted_id),
        "meter_id": meter_id,
        "area_zone": assigned_zone,
        "message": f"Meter '{meter_id}' registered successfully on {assigned_zone}.",
    }


@router.get("/registry/{meter_id}/details")
def get_meter_profile(meter_id: str):
    profile = meters_collection.find_one({"meter_id": meter_id})
    readings = list(
        meter_readings.find({"meter_id": meter_id})
        .sort("created_at", -1)
        .limit(20)
    )

    inv = investigations.find_one({"meter_id": meter_id})
    current_status = inv.get("status") if inv else "none"

    for r in readings:
        r["id"] = str(r.pop("_id"))
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = r["created_at"].isoformat()
        r["audit_status"] = r.get("audit_status") or current_status

    return {
        "profile": {
            "meter_id": meter_id,
            "consumer_name": profile.get("consumer_name", "Unassigned") if profile else "Unassigned",
            "sanctioned_load_kw": profile.get("sanctioned_load_kw", "—") if profile else "—",
            "area_zone": profile.get("area_zone", "General Substation") if profile else "General Substation",
            "status": profile.get("status", "Active") if profile else "Active",
            "current_audit_status": current_status,
        },
        "readings": readings,
    }


# --------------------------------------------------
# Predictions & Inference Pipeline
# --------------------------------------------------

def process_single_prediction(reading: MLPredictionRequest):
    meter_id = (reading.meter_id or "").strip()
    if not meter_id:
        meter_id = f"MTR-{datetime.now(timezone.utc).strftime('%H%M%S')}"

    features = reading.model_dump(exclude={"meter_id"})

    try:
        result = predict_meter(features)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(error)}") from error

    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="ML prediction returned an invalid null response.")

    created_at = datetime.now(timezone.utc)
    assigned_zone = random.choice(DEFAULT_FEEDER_ZONES)

    try:
        meters_collection.update_one(
            {"meter_id": meter_id},
            {
                "$setOnInsert": {
                    "meter_id": meter_id,
                    "consumer_name": f"Consumer ({meter_id})",
                    "sanctioned_load_kw": 5.0,
                    "meter_type": "Three Phase Smart",
                    "area_zone": assigned_zone,
                    "status": "Active",
                    "created_at": created_at,
                }
            },
            upsert=True,
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Database error writing to meters registry: {str(error)}") from error

    existing_inv = investigations.find_one({"meter_id": meter_id})
    is_suspicious_flag = bool(result.get("is_suspicious")) or (float(result.get("risk_score", 0.0)) >= 50.0)

    if existing_inv is not None:
        initial_audit_status = existing_inv.get("status", "pending")
    else:
        initial_audit_status = "pending" if is_suspicious_flag else "none"

    risk_score = float(result.get("risk_score", 0.0))
    threshold = float(result.get("threshold", 0.5))
    severity = str(result.get("severity", "Low"))
    prediction = str(result.get("prediction", "normal_pattern"))
    reasons = result.get("reasons") or []
    warning = result.get("warning")

    document = {
        "meter_id": meter_id,
        "previous_month_kwh": reading.previous_month_kwh,
        "current_month_kwh": reading.current_month_kwh,
        "avg_daily_kwh": reading.avg_daily_kwh,
        "usage_change_pct": reading.usage_change_pct,
        "night_usage_ratio": reading.night_usage_ratio,
        "weekend_usage_ratio": reading.weekend_usage_ratio,
        "zero_reading_days": reading.zero_reading_days,
        "risk_score": risk_score,
        "threshold": threshold,
        "severity": severity,
        "prediction": prediction,
        "reasons": reasons,
        "warning": warning,
        "audit_status": initial_audit_status,
        "created_at": created_at,
    }

    try:
        saved = meter_readings.insert_one(document)
        reading_id = str(saved.inserted_id)

        if is_suspicious_flag:
            existing_status = existing_inv.get("status", "pending") if existing_inv is not None else "pending"
            investigations.update_one(
                {"meter_id": meter_id},
                {
                    "$set": {
                        "meter_id": meter_id,
                        "latest_reading_id": reading_id,
                        "risk_score": risk_score,
                        "severity": severity,
                        "reasons": reasons,
                        "status": existing_status,
                        "updated_at": created_at,
                    },
                    "$setOnInsert": {
                        "created_at": created_at,
                        "notes": "",
                    },
                },
                upsert=True,
            )
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Database error saving reading: {str(error)}") from error

    return {
        "id": reading_id,
        "meter_id": meter_id,
        "risk_score": risk_score,
        "threshold": threshold,
        "severity": severity,
        "is_suspicious": is_suspicious_flag,
        "prediction": prediction,
        "reasons": reasons,
        "warning": warning,
        "audit_status": initial_audit_status,
        "created_at": created_at.isoformat(),
        "previous_month_kwh": reading.previous_month_kwh,
        "current_month_kwh": reading.current_month_kwh,
        "avg_daily_kwh": reading.avg_daily_kwh,
        "usage_change_pct": reading.usage_change_pct,
        "zero_reading_days": reading.zero_reading_days,
    }


@router.post("/predict")
def predict_meter_reading(reading: MLPredictionRequest):
    return process_single_prediction(reading)


@router.post("/predict/batch")
async def batch_predict_csv(file: UploadFile = File(...)):
    """Upload a CSV file containing telemetry batches to run bulk inference."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV file: {str(e)}")

    required_cols = ["previous_month_kwh", "current_month_kwh"]
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(status_code=422, detail=f"Missing required column in CSV: '{col}'")

    processed = []
    for idx, row in df.iterrows():
        mid = str(row.get("meter_id") or f"BATCH-{idx+1}").strip()
        prev = float(row.get("previous_month_kwh", 0))
        curr = float(row.get("current_month_kwh", 0))
        daily = float(row.get("avg_daily_kwh", curr / 30 if curr > 0 else 0))
        change = float(row.get("usage_change_pct", ((curr - prev) / prev * 100) if prev > 0 else 0))

        night = min(1.5, max(0.0, float(row.get("night_usage_ratio", 0.30))))
        weekend = min(2.0, max(0.0, float(row.get("weekend_usage_ratio", 0.95))))
        zero_days = max(0, int(row.get("zero_reading_days", 0)))

        req = MLPredictionRequest(
            meter_id=mid,
            previous_month_kwh=prev,
            current_month_kwh=curr,
            avg_daily_kwh=round(daily, 2),
            usage_change_pct=round(change, 2),
            night_usage_ratio=night,
            weekend_usage_ratio=weekend,
            zero_reading_days=zero_days,
        )
        res = process_single_prediction(req)
        processed.append(res)

    return {
        "message": f"Successfully processed {len(processed)} meters from CSV file.",
        "count": len(processed),
        "results": processed,
    }


@router.post("/predict/pdf")
async def predict_from_bill_pdf(file: UploadFile = File(...)):
    """Extract billing telemetry from an electricity bill PDF and run ML anomaly analysis."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF invoice files are accepted.")

    contents = await file.read()
    try:
        reader = PdfReader(io.BytesIO(contents))
        extracted_text = ""
        for page in reader.pages:
            extracted_text += (page.extract_text() or "") + "\n"
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to read PDF file: {str(e)}")

    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in PDF. Ensure it is a text-based digital bill.")

    # 1. Extract Meter ID
    meter_id_match = re.search(
        r"(?:Meter\s*(?:No|ID|#)|CA\s*No|Consumer\s*No)[\s:\n|]*([A-Za-z0-9\-]+)",
        extracted_text,
        re.IGNORECASE,
    )
    meter_id = meter_id_match.group(1).strip() if meter_id_match else f"BILL-{datetime.now(timezone.utc).strftime('%H%M%S')}"

    # 2. Extract Units Consumed (Current Month kWh)
    units_match = re.search(
        r"(?:Units\s*Consumed\s*\(kWh\)|Billed\s*Units)[\s:\n|]*([0-9,]+(?:\.[0-9]+)?)",
        extracted_text,
        re.IGNORECASE,
    )
    if units_match:
        curr_kwh = float(units_match.group(1).replace(",", ""))
    else:
        curr_kwh_match = re.search(
            r"(?:Current\s*Reading)[\s:\n|]*([0-9,]+(?:\.[0-9]+)?)",
            extracted_text,
            re.IGNORECASE,
        )
        curr_kwh = float(curr_kwh_match.group(1).replace(",", "")) if curr_kwh_match else 237.0

    # 3. Extract Previous Month Units (from 6-Month history or baseline difference)
    history_match = re.findall(r"\b([0-9]{2,4})\b", extracted_text)
    if history_match and len(history_match) >= 2:
        prev_kwh = float(history_match[-2])
    else:
        prev_kwh = round(curr_kwh * 1.10, 1)

    daily_kwh = round(curr_kwh / 30.0, 2)
    usage_change = round(((curr_kwh - prev_kwh) / prev_kwh) * 100, 2) if prev_kwh > 0 else 0.0

    req = MLPredictionRequest(
        meter_id=meter_id,
        previous_month_kwh=prev_kwh,
        current_month_kwh=curr_kwh,
        avg_daily_kwh=daily_kwh,
        usage_change_pct=usage_change,
        night_usage_ratio=0.30,
        weekend_usage_ratio=0.95,
        zero_reading_days=4 if usage_change < -50 else 0,
    )

    evaluation = process_single_prediction(req)

    return {
        "message": f"Successfully parsed bill PDF for meter '{meter_id}'.",
        "extracted_data": {
            "meter_id": meter_id,
            "previous_month_kwh": prev_kwh,
            "current_month_kwh": curr_kwh,
            "usage_change_pct": usage_change,
        },
        "evaluation": evaluation,
    }


@router.get("/history")
def get_meter_history(limit: int = 50):
    limit = max(1, min(limit, 100))
    try:
        records = list(meter_readings.find().sort("created_at", -1).limit(limit))
    except Exception as error:
        raise HTTPException(status_code=503, detail="Unable to retrieve history from MongoDB.") from error

    inv_map = {
        item["meter_id"]: item.get("status", "pending")
        for item in investigations.find({}, {"meter_id": 1, "status": 1})
    }

    for record in records:
        record["id"] = str(record.pop("_id"))
        if isinstance(record.get("created_at"), datetime):
            record["created_at"] = record["created_at"].isoformat()

        mid = record.get("meter_id")
        record["audit_status"] = record.get("audit_status") or inv_map.get(mid, "none")

    return {"count": len(records), "records": records}


# --------------------------------------------------
# History Deletion Endpoints
# --------------------------------------------------

@router.delete("/history/record/{reading_id}")
def delete_single_reading(reading_id: str):
    try:
        res = meter_readings.delete_one({"_id": ObjectId(reading_id)})
        if res.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Reading record not found.")
        return {"message": "Reading removed successfully.", "deleted_count": res.deleted_count}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to delete reading: {str(error)}") from error


@router.delete("/history/meter/{meter_id}")
def delete_meter_history(meter_id: str):
    try:
        res_readings = meter_readings.delete_many({"meter_id": meter_id})
        res_inv = investigations.delete_many({"meter_id": meter_id})
        return {
            "message": f"Cleared history for {meter_id}.",
            "deleted_readings": res_readings.deleted_count,
            "deleted_investigations": res_inv.deleted_count,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to delete meter history: {str(error)}") from error


@router.delete("/history")
def clear_all_history():
    try:
        res_readings = meter_readings.delete_many({})
        res_inv = investigations.delete_many({})
        return {
            "message": "All meter history and investigations purged successfully.",
            "deleted_readings": res_readings.deleted_count,
            "deleted_investigations": res_inv.deleted_count,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(error)}") from error


@router.get("/investigations")
def get_investigations():
    try:
        count = investigations.count_documents({})
        if count == 0:
            suspicious_readings = list(
                meter_readings.find({
                    "$or": [
                        {"prediction": "suspicious_pattern"},
                        {"prediction": "Suspicious"},
                        {"risk_score": {"$gte": 50}}
                    ]
                }).sort("created_at", -1)
            )
            for r in suspicious_readings:
                mid = r.get("meter_id", "Unknown")
                score = r.get("risk_score", 0)
                sev = "High" if score >= 70 else "Medium"
                investigations.update_one(
                    {"meter_id": mid},
                    {
                        "$set": {
                            "meter_id": mid,
                            "latest_reading_id": str(r["_id"]),
                            "risk_score": score,
                            "severity": r.get("severity", sev),
                            "reasons": r.get("reasons", ["Significant drop in consumption detected"]),
                            "status": "pending",
                            "updated_at": r.get("created_at", datetime.now(timezone.utc)),
                        },
                        "$setOnInsert": {
                            "created_at": r.get("created_at", datetime.now(timezone.utc)),
                            "notes": "",
                        },
                    },
                    upsert=True,
                )

        items = list(investigations.find().sort("updated_at", -1))
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve investigations from MongoDB.",
        ) from error

    for item in items:
        item["id"] = str(item.pop("_id"))
        if isinstance(item.get("updated_at"), datetime):
            item["updated_at"] = item["updated_at"].isoformat()
        if isinstance(item.get("created_at"), datetime):
            item["created_at"] = item["created_at"].isoformat()

    return {"investigations": items}


@router.patch("/investigations/{meter_id}/status")
def update_investigation_status(meter_id: str, body: UpdateInvestigationStatusRequest):
    updated_at = datetime.now(timezone.utc)

    result = investigations.update_one(
        {"meter_id": meter_id},
        {"$set": {"status": body.status, "notes": body.notes, "updated_at": updated_at}},
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"No investigation found for meter {meter_id}")

    meter_readings.update_many(
        {"meter_id": meter_id},
        {"$set": {"audit_status": body.status, "audit_updated_at": updated_at}}
    )

    return {
        "message": f"Audit status for meter {meter_id} recorded in history.",
        "status": body.status,
    }


# --------------------------------------------------
# Feeder Energy Balance & Connected Meters Analytics
# --------------------------------------------------

@router.get("/analytics/feeder-balance")
def get_feeder_balance(tariff_per_kwh: float = 6.50):
    try:
        meters = list(meters_collection.find())
        readings = list(meter_readings.find().sort("created_at", -1))

        meters_dict = {m.get("meter_id"): m for m in meters}

        latest_readings = {}
        for r in readings:
            mid = r.get("meter_id")
            if mid and mid not in latest_readings:
                latest_readings[mid] = r

        all_meter_ids = list(set(list(meters_dict.keys()) + list(latest_readings.keys())))

        feeder_groups = {}
        for idx, mid in enumerate(all_meter_ids):
            m_profile = meters_dict.get(mid, {})
            reading = latest_readings.get(mid, {})

            zone = m_profile.get("area_zone")
            if not zone or zone in ["General Feeder", "General Substation"]:
                zone = DEFAULT_FEEDER_ZONES[idx % len(DEFAULT_FEEDER_ZONES)]

            consumer_name = m_profile.get("consumer_name", f"Consumer ({mid})")
            curr_kwh = float(reading.get("current_month_kwh", 0))
            is_theft = (
                reading.get("prediction") in ["suspicious_pattern", "Suspicious"]
                or float(reading.get("risk_score", 0)) >= 50
            )

            if zone not in feeder_groups:
                feeder_groups[zone] = {
                    "zone": zone,
                    "meter_count": 0,
                    "connected_meters": [],
                    "billed_consumption_kwh": 0.0,
                    "suspicious_meters": 0,
                    "estimated_stolen_kwh": 0.0,
                }

            feeder_groups[zone]["meter_count"] += 1
            feeder_groups[zone]["connected_meters"].append({
                "meter_id": mid,
                "consumer_name": consumer_name,
                "is_suspicious": is_theft,
                "risk_score": float(reading.get("risk_score", 0)),
            })
            feeder_groups[zone]["billed_consumption_kwh"] += curr_kwh

            if is_theft:
                feeder_groups[zone]["suspicious_meters"] += 1
                prev_kwh = float(reading.get("previous_month_kwh", curr_kwh))
                diff = max(0.0, prev_kwh - curr_kwh)
                feeder_groups[zone]["estimated_stolen_kwh"] += diff

        summary = []
        for zone, data in feeder_groups.items():
            billed = data["billed_consumption_kwh"]
            stolen = data["estimated_stolen_kwh"]
            tech_loss = billed * 0.08
            supplied = billed + stolen + tech_loss

            revenue_at_risk = stolen * tariff_per_kwh
            loss_percentage = (stolen / supplied * 100) if supplied > 0 else 0.0

            summary.append({
                "feeder_zone": zone,
                "meters": data["meter_count"],
                "connected_meters": data["connected_meters"],
                "supplied_kwh": round(supplied, 1),
                "billed_kwh": round(billed, 1),
                "stolen_kwh": round(stolen, 1),
                "non_technical_loss_pct": round(loss_percentage, 1),
                "revenue_at_risk_inr": round(revenue_at_risk, 2),
                "suspicious_meters": data["suspicious_meters"],
            })

        return {"tariff_rate_inr": tariff_per_kwh, "feeders": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feeder balance calculation error: {str(e)}")