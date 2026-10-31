# backend/schemas/analyses.py
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field


class EngineOut(BaseModel):
    name: str
    label: str
    available: bool
    reason: Optional[str] = None


class HealthOut(BaseModel):
    status: str
    version: str
    engines: list[EngineOut]


class DemoCaseOut(BaseModel):
    case_id: str


class DemoRequest(BaseModel):
    engine: str = Field(default="classique", max_length=32)


class AnalysisOut(BaseModel):
    id: str
    source: Literal["upload", "demo"]
    case_id: Optional[str]
    engine: str
    status: Literal["queued", "running", "done", "failed"]
    step: Optional[str]
    progress: float
    error: Optional[str]
    summary: Optional[dict]
    owner: Optional[str] = None
    created_at: str
    finished_at: Optional[str]

    @classmethod
    def from_row(cls, row) -> "AnalysisOut":
        keys = row.keys()
        return cls(id=row["id"], source=row["source"], case_id=row["case_id"], engine=row["engine"],
                   status=row["status"], step=row["step"], progress=row["progress"],
                   error=row["error"],
                   summary=json.loads(row["summary_json"]) if row["summary_json"] else None,
                   owner=row["username"] if "username" in keys else None,
                   created_at=row["created_at"], finished_at=row["finished_at"])


class SubmitOut(BaseModel):
    analysis_id: str
