"""Tests for Multi-tenant isolation and Role-Based Access Control."""

from signalrag.core.security import AccessController, AccessControlPolicy, SecurityContext
from signalrag.models.chunk import Chunk, ChunkMetadata


def test_tenant_boundary_isolation():
    controller = AccessController()
    ctx = SecurityContext(user_id="alice", tenant_id="org_alpha", roles=["reader"])
    policy_alpha = AccessControlPolicy(tenant_id="org_alpha")
    policy_beta = AccessControlPolicy(tenant_id="org_beta")

    assert controller.is_authorized(ctx, policy_alpha) is True
    assert controller.is_authorized(ctx, policy_beta) is False


def test_role_hierarchy():
    controller = AccessController()
    reader_ctx = SecurityContext(user_id="bob", tenant_id="tenant1", roles=["reader"])
    admin_ctx = SecurityContext(user_id="carol", tenant_id="tenant1", roles=["admin"])
    admin_policy = AccessControlPolicy(tenant_id="tenant1", min_role="admin")

    assert controller.is_authorized(reader_ctx, admin_policy) is False
    assert controller.is_authorized(admin_ctx, admin_policy) is True


def test_filter_chunks_rbac():
    controller = AccessController()
    ctx = SecurityContext(user_id="dave", tenant_id="tenant_a", roles=["reader"], groups=["eng"])

    c1 = Chunk(
        id="c1",
        document_id="d1",
        text="Public tenant A doc",
        metadata=ChunkMetadata(
            document_id="d1",
            chunk_index=0,
            source="s1",
            extra={"tenant_id": "tenant_a", "allowed_groups": ["eng"]},
        ),
    )
    c2 = Chunk(
        id="c2",
        document_id="d2",
        text="Tenant B doc",
        metadata=ChunkMetadata(
            document_id="d2", chunk_index=0, source="s2", extra={"tenant_id": "tenant_b"}
        ),
    )

    filtered = controller.filter_chunks(ctx, [c1, c2])
    assert len(filtered) == 1
    assert filtered[0].id == "c1"
