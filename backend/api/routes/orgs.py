"""
backend/api/routes/orgs.py — Workspaces + members API (Phase 3).

  GET    /workspaces
  POST   /workspaces
  GET    /workspaces/{id}/members
  POST   /workspaces/{id}/members
  DELETE /workspaces/{id}/members/{user_id}
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth import org_store
from auth.store import get_user_by_email, get_user_by_id

from ..deps import bootstrap_user_workspace, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["orgs"])


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class MemberAdd(BaseModel):
    email: Optional[str] = Field(default=None, max_length=256)
    user_id: Optional[str] = Field(default=None, max_length=64)
    role: Literal["owner", "analyst"] = "analyst"


@router.get("/workspaces")
async def list_workspaces(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    if user_id.startswith("guest-"):
        return {"workspaces": []}
    await asyncio.to_thread(bootstrap_user_workspace, user_id)
    items = await asyncio.to_thread(org_store.list_workspaces, user_id)
    return {"workspaces": [w.to_dict() for w in items]}


@router.post("/workspaces", status_code=status.HTTP_201_CREATED)
async def create_workspace(
    body: WorkspaceCreate,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    if user_id.startswith("guest-"):
        raise HTTPException(status_code=403, detail="Guests cannot create workspaces")
    try:
        ws = await asyncio.to_thread(org_store.create_workspace, user_id, name=body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("workspace.created user=%s id=%s", user_id, ws.workspace_id)
    return ws.to_dict()


@router.get("/workspaces/{workspace_id}/members")
async def list_members(
    workspace_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        await asyncio.to_thread(
            org_store.require_role,
            current_user["user_id"],
            workspace_id,
            min_role="analyst",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"members": [m.to_dict() for m in await asyncio.to_thread(
        org_store.list_members,
        workspace_id,
    )]}


@router.post("/workspaces/{workspace_id}/members", status_code=status.HTTP_201_CREATED)
async def add_member(
    workspace_id: str,
    body: MemberAdd,
    current_user: dict = Depends(get_current_user),
):
    try:
        await asyncio.to_thread(
            org_store.require_role,
            current_user["user_id"],
            workspace_id,
            min_role="owner",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    target_id = (body.user_id or "").strip()
    if not target_id and body.email:
        user = await asyncio.to_thread(get_user_by_email, body.email.strip().lower())
        if not user:
            raise HTTPException(status_code=404, detail="User not found for that email")
        target_id = user.user_id
    if not target_id:
        raise HTTPException(status_code=400, detail="email or user_id required")
    if target_id.startswith("guest-"):
        raise HTTPException(status_code=400, detail="Cannot add guests to workspaces")
    if not await asyncio.to_thread(get_user_by_id, target_id):
        raise HTTPException(status_code=404, detail="User not found")

    try:
        member = await asyncio.to_thread(
            org_store.add_member,
            workspace_id,
            user_id=target_id,
            role=body.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return member.to_dict()


@router.delete(
    "/workspaces/{workspace_id}/members/{member_user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_member(
    workspace_id: str,
    member_user_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        await asyncio.to_thread(
            org_store.require_role,
            current_user["user_id"],
            workspace_id,
            min_role="owner",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        ok = await asyncio.to_thread(org_store.remove_member, workspace_id, member_user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=404, detail="Member not found")
    return None
