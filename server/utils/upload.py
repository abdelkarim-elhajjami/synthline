"""Shared file upload parsing for multipart and raw-body requests."""
from typing import Tuple

from fastapi import HTTPException, Request


async def parse_upload(request: Request) -> Tuple[str, bytes]:
    """Extract (filename, payload) from a multipart or raw-body upload.

    Raises HTTPException on missing filename or empty payload.
    """
    filename = None
    payload = b""
    content_type = (request.headers.get("content-type") or "").lower()

    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse multipart upload: {exc}")

        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="Multipart form must include a `file` field.")

        filename = getattr(upload, "filename", None)
        payload = await upload.read()
    else:
        filename = request.headers.get("x-filename")
        payload = await request.body()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename must be provided via `filename` in multipart form or `x-filename` header.",
        )

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return filename, payload
