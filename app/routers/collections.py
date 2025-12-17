"""
routers/collections.py — collection (namespace) management
"""

import uuid
from fastapi import APIRouter, HTTPException
from app.database import query as db_query, execute
from app.models.schemas import CollectionCreate, CollectionResponse

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=201)
def create_collection(req: CollectionCreate):
    existing = db_query("SELECT id FROM collections WHERE name = %s", (req.name,))
    if existing:
        raise HTTPException(status_code=409, detail=f"Collection '{req.name}' already exists")

    cid = str(uuid.uuid4())
    execute(
        "INSERT INTO collections (id, name, description) VALUES (%s, %s, %s)",
        (cid, req.name, req.description),
    )
    rows = db_query("SELECT * FROM collections WHERE id = %s", (cid,))
    r = rows[0]
    return CollectionResponse(id=str(r["id"]), name=r["name"],
                               description=r["description"], created_at=r["created_at"])


@router.get("", response_model=list[CollectionResponse])
def list_collections():
    rows = db_query("SELECT * FROM collections ORDER BY created_at DESC")
    return [CollectionResponse(id=str(r["id"]), name=r["name"],
                                description=r["description"], created_at=r["created_at"])
            for r in rows]


@router.delete("/{name}", status_code=204)
def delete_collection(name: str):
    if name == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default collection")
    rows = db_query("SELECT id FROM collections WHERE name = %s", (name,))
    if not rows:
        raise HTTPException(status_code=404, detail="Collection not found")
    execute("DELETE FROM collections WHERE name = %s", (name,))
