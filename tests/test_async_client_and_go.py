"""Tests for AsyncSignalRAGClient and Go CLI assets."""

import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from signalrag.api.app import app
from signalrag.client import AsyncSignalRAGClient


def test_async_python_client():
    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            async with AsyncSignalRAGClient(client=http_client) as client:
                doc_res = await client.add_document(
                    content="Asynchronous clients scale high-concurrency event loops.",
                    metadata={"tag": "async"},
                )
                assert doc_res.document_id is not None

                idx_res = await client.index()
                assert idx_res.status == "success"

                q_res = await client.query("How do async clients scale?", top_k=2)
                assert q_res.answer is not None

    asyncio.run(_run())


def test_go_cli_source_exists():
    go_cli = Path("clients/go/cmd/signalrag-cli/main.go")
    assert go_cli.exists()
    content = go_cli.read_text()
    assert "package main" in content
    assert "signalrag.NewClient" in content
