from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConfigPayload(BaseModel):
    """Payload for saving a SUMA YAML configuration."""

    path: str | None = None
    config: dict[str, Any] | None = None
    raw_yaml: str | None = None


class ValidationPayload(BaseModel):
    """Payload for validating structured or raw YAML configuration content."""

    config: dict[str, Any] | None = None
    raw_yaml: str | None = None


class ConfigCreatePayload(BaseModel):
    """Payload for creating a config from a clean starter or an existing source."""

    path: str
    source_path: str | None = None


class ConfigDeletePayload(BaseModel):
    """Payload for deleting a repository-relative config file."""

    path: str


class JobCreatePayload(BaseModel):
    """Payload for launching a registered SUMA workflow as a managed job."""

    workflow_id: str = Field(..., description="Workflow identifier from /api/workflows")
    payload: dict[str, Any] = Field(default_factory=dict)


class CitySpeedLimitUpdatePayload(BaseModel):
    """Payload for bulk-editing OSM way speed-limit tags in an extracted city."""

    way_ids: list[str] = Field(default_factory=list)
    speed_kph: float


class CityWaySelectionPayload(BaseModel):
    """Payload containing selected OSM way identifiers."""

    way_ids: list[str] = Field(default_factory=list)


class ResultDeletePayload(BaseModel):
    """Payload for deleting a completed SUMA result run folder."""

    path: str
