"""Multi-tenant partitioning, ACL evaluation, and Role-Based Access Control (RBAC)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from signalrag.models.chunk import Chunk


class SecurityContext(BaseModel):
    """Identity and authorization context for the requesting subject."""

    user_id: str
    tenant_id: str = "default"
    roles: list[str] = Field(default_factory=lambda: ["reader"])
    groups: list[str] = Field(default_factory=list)


class AccessControlPolicy(BaseModel):
    """Access policy attached to documents and chunks."""

    tenant_id: str = "default"
    allowed_groups: list[str] = Field(default_factory=list)
    min_role: str = "reader"  # reader, contributor, admin


class AccessController:
    """Enforces multi-tenant isolation and role/group authorization."""

    ROLE_HIERARCHY = {
        "reader": 1,
        "contributor": 2,
        "admin": 3,
    }

    def is_authorized(
        self,
        context: SecurityContext,
        policy: AccessControlPolicy,
    ) -> bool:
        """Evaluate if security context satisfies tenant, role, and group policies."""
        # 1. Tenant boundary must strictly match
        if context.tenant_id != policy.tenant_id:
            return False

        # 2. Role hierarchy check
        user_max_role_level = max(
            [self.ROLE_HIERARCHY.get(r.lower(), 1) for r in context.roles] or [1]
        )
        required_level = self.ROLE_HIERARCHY.get(policy.min_role.lower(), 1)
        if user_max_role_level < required_level:
            return False

        # 3. Group check (if policy specifies allowed_groups, user must belong to at least one)
        if policy.allowed_groups:
            user_groups = set(context.groups)
            if not user_groups.intersection(set(policy.allowed_groups)):
                # Admins bypass group restrictions within their own tenant
                if user_max_role_level < self.ROLE_HIERARCHY["admin"]:
                    return False

        return True

    def filter_chunks(
        self,
        context: SecurityContext,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Filter chunk candidates down to only those authorized for the security context."""
        authorized: list[Chunk] = []
        for c in chunks:
            extra = c.metadata.extra if c.metadata else {}
            chunk_tenant = extra.get("tenant_id", "default")
            chunk_groups = extra.get("allowed_groups", [])
            chunk_min_role = extra.get("min_role", "reader")

            policy = AccessControlPolicy(
                tenant_id=chunk_tenant,
                allowed_groups=chunk_groups,
                min_role=chunk_min_role,
            )
            if self.is_authorized(context, policy):
                authorized.append(c)

        return authorized
