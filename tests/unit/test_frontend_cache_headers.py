from starlette.responses import Response

from api import (
    HTML_CACHE_CONTROL,
    STATIC_ASSET_CACHE_CONTROL,
    _apply_frontend_cache_headers,
)


def test_frontend_html_is_not_cached():
    response = Response()

    _apply_frontend_cache_headers("/", response)

    assert response.headers["Cache-Control"] == HTML_CACHE_CONTROL


def test_existing_next_static_assets_are_immutable():
    response = Response(status_code=200)

    _apply_frontend_cache_headers("/_next/static/chunks/app/page.js", response)

    assert response.headers["Cache-Control"] == STATIC_ASSET_CACHE_CONTROL


def test_missing_next_static_assets_are_not_cached():
    response = Response(status_code=404)

    _apply_frontend_cache_headers("/_next/static/chunks/app/old-page.js", response)

    assert response.headers["Cache-Control"] == HTML_CACHE_CONTROL


def test_api_responses_are_unchanged():
    response = Response()

    _apply_frontend_cache_headers("/api/health", response)

    assert "Cache-Control" not in response.headers
