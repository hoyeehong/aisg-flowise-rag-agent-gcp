"""
Request authentication and per-request tenant resolution.

This exists because row-level security is only as trustworthy as the tenant identifier
handed to it. A policy that filters on ``app.tenant_id`` enforces perfectly against
whatever value it is given, so if that value comes from a request header or body, a
caller simply names someone else's tenant and the database obliges. The tenant must
come from a credential the service verifies, which is what this module does.

Symmetric (HS*) signatures only. Verifying RS256 against a JWKS endpoint is the
production shape for a real identity provider, and it is deliberately *not* implemented
here rather than half-implemented: it needs key fetching, caching, rotation and
`kid` selection, none of which are testable without an IdP. Configuring an asymmetric
algorithm is rejected at startup instead of silently doing something weaker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jwt
from fastapi import HTTPException, status

from ..tenancy import is_valid_tenant_id

if TYPE_CHECKING:
    from ..config import Settings

# Symmetric only -- see the module docstring. The value is the minimum secret length
# in bytes: RFC 7518 section 3.2 requires an HMAC key at least as long as the hash
# output, because a shorter one can be brute-forced offline from a single valid token.
# PyJWT only warns about this, so the check is enforced here.
SUPPORTED_ALGORITHMS = {"HS256": 32, "HS384": 48, "HS512": 64}


@dataclass(frozen=True)
class Principal:
    """Who is making this request, and which tenant's data they may touch."""

    tenant_id: str
    subject: str
    # "jwt" when a verified token supplied the tenant, "anonymous" when the service is
    # deliberately configured single-tenant. Surfaced so a response can never leave a
    # reader guessing which of the two produced it.
    mode: str


class AuthConfigurationError(RuntimeError):
    """Raised at startup for a configuration that cannot be enforced safely."""


class TenantResolver:
    """
    Resolves a request's principal from its Authorization header.

    Three modes, decided by configuration at construction:

    * ``jwt`` -- a signing secret is configured; every request must present a valid
      bearer token and the tenant comes from a claim.
    * ``anonymous`` -- no secret, and ``allow_anonymous_tenant`` is set. The configured
      ``tenant_id`` is used. This is the local-development and CI shape, and it is an
      explicit opt-in for the same reason the hashing embedder is: a missing credential
      should not quietly become an accepted configuration.
    * ``unconfigured`` -- no secret and no opt-in. Tenant-scoped requests are refused
      with 503, because the service cannot establish who is asking and answering
      anyway would serve one tenant's data under no authority at all.
    """

    def __init__(self, settings: Settings) -> None:
        secret = settings.jwt_secret.get_secret_value()
        self._secret = secret
        self._algorithm = settings.jwt_algorithm
        self._audience = settings.jwt_audience
        self._issuer = settings.jwt_issuer
        self._claim = settings.jwt_tenant_claim
        self._fallback_tenant = settings.tenant_id

        if secret:
            minimum = SUPPORTED_ALGORITHMS.get(self._algorithm)
            if minimum is None:
                raise AuthConfigurationError(
                    f"jwt_algorithm {self._algorithm!r} is not supported; "
                    f"expected one of {sorted(SUPPORTED_ALGORITHMS)}. "
                    "Asymmetric verification against a JWKS endpoint is not implemented."
                )
            if len(secret.encode("utf-8")) < minimum:
                # Refused rather than warned about. A signing secret that is too short
                # makes every other check here decorative, and the failure is silent:
                # tokens verify normally right up until someone recovers the key.
                raise AuthConfigurationError(
                    f"jwt_secret is too short for {self._algorithm}: "
                    f"{len(secret.encode('utf-8'))} bytes, minimum {minimum} "
                    "(RFC 7518 section 3.2)"
                )
            self.mode = "jwt"
        elif settings.allow_anonymous_tenant:
            self.mode = "anonymous"
        else:
            self.mode = "unconfigured"

    @property
    def enforcing(self) -> bool:
        """True when a verified credential is required to reach tenant-scoped data."""
        return self.mode == "jwt"

    @property
    def usable(self) -> bool:
        """True when tenant-scoped requests can be served at all."""
        return self.mode in {"jwt", "anonymous"}

    def resolve(self, authorization: str | None) -> Principal:
        if self.mode == "unconfigured":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "tenant resolution is not configured: set AGENT_JWT_SECRET, or "
                    "AGENT_ALLOW_ANONYMOUS_TENANT=true to accept the configured tenant"
                ),
            )
        if self.mode == "anonymous":
            return Principal(tenant_id=self._fallback_tenant, subject="anonymous", mode="anonymous")

        token = self._bearer(authorization)
        claims = self._verify(token)
        return Principal(
            tenant_id=self._tenant_from(claims),
            subject=str(claims.get("sub") or "unknown"),
            mode="jwt",
        )

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _unauthorised(detail: str) -> HTTPException:
        # RFC 6750: a 401 on a bearer-protected resource carries the challenge, which
        # is what tells a client to retry with a credential rather than give up.
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _bearer(self, authorization: str | None) -> str:
        if not authorization:
            raise self._unauthorised("missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise self._unauthorised("expected an 'Authorization: Bearer <token>' header")
        return token.strip()

    def _verify(self, token: str) -> dict[str, Any]:
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                self._secret,
                # An explicit allowlist. Never trust the token's own `alg` header: that
                # is what makes algorithm-confusion and `alg: none` attacks work.
                algorithms=[self._algorithm],
                audience=self._audience or None,
                issuer=self._issuer or None,
                options={
                    # A token with no expiry is a permanent credential. Require one
                    # rather than accepting whatever the issuer felt like sending.
                    "require": ["exp"],
                    # PyJWT rejects a token carrying `aud` when no audience is expected,
                    # so only verify it when one is actually configured.
                    "verify_aud": bool(self._audience),
                    "verify_iss": bool(self._issuer),
                },
            )
        except jwt.ExpiredSignatureError as exc:
            raise self._unauthorised("token has expired") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise self._unauthorised(f"token is missing a required claim: {exc.claim}") from exc
        except jwt.InvalidTokenError as exc:
            # Deliberately not echoing the library's message: it distinguishes "bad
            # signature" from "malformed", which tells an attacker which half to fix.
            raise self._unauthorised("token is not valid") from exc
        return claims

    def _tenant_from(self, claims: dict[str, Any]) -> str:
        raw = claims.get(self._claim)
        if not is_valid_tenant_id(raw):
            # 403, not 401: the credential is genuine, it just does not authorise any
            # tenant. Retrying with the same token will not help.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"token has no usable {self._claim!r} claim",
            )
        return raw
