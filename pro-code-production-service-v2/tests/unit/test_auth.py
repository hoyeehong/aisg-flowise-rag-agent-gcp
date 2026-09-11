"""
Tenant resolution and bearer-token verification.

The security-relevant cases here are the negative ones. A resolver that accepts a
valid token is easy; what matters is that it rejects a token signed with the wrong
key, a token with no expiry, a token whose `alg` header says `none`, and a token that
carries no tenant. Each of those has been a real-world JWT bypass.
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import HTTPException

from digital_economy_agent.api.auth import AuthConfigurationError, TenantResolver
from digital_economy_agent.config import Settings

SECRET = "unit-test-signing-secret-32bytes!!"  # >= 32 bytes, as HS256 requires


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "groq_api_key": "k",
        "model_chain": "primary",
        "tenant_id": "configured-tenant",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _token(secret: str = SECRET, algorithm: str = "HS256", **claims: object) -> str:
    payload: dict[str, object] = {
        "sub": "user-1",
        "tenant_id": "acme",
        "exp": int(time.time()) + 300,
    }
    payload.update(claims)
    return jwt.encode(payload, secret, algorithm=algorithm)


# --- mode selection --------------------------------------------------------


def test_unconfigured_refuses_with_503() -> None:
    """
    No secret and no explicit opt-in must not silently serve one tenant to everyone.

    503 rather than 401: the caller did nothing wrong, the deployment is incomplete.
    A 401 would invite them to go and find a credential that cannot exist yet.
    """
    resolver = TenantResolver(_settings())
    assert resolver.mode == "unconfigured"
    assert resolver.usable is False
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(None)
    assert exc.value.status_code == 503


def test_anonymous_mode_uses_the_configured_tenant() -> None:
    resolver = TenantResolver(_settings(allow_anonymous_tenant=True))
    assert resolver.mode == "anonymous"
    assert resolver.enforcing is False
    principal = resolver.resolve(None)
    assert principal.tenant_id == "configured-tenant"
    assert principal.mode == "anonymous"


def test_a_secret_takes_precedence_over_the_anonymous_opt_in() -> None:
    """Opting in to anonymous access must not weaken a configured secret."""
    resolver = TenantResolver(_settings(jwt_secret=SECRET, allow_anonymous_tenant=True))
    assert resolver.enforcing is True
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(None)
    assert exc.value.status_code == 401


def test_asymmetric_algorithm_is_rejected_at_construction() -> None:
    """
    RS256 is not implemented, so it must fail loudly at startup.

    Accepting the setting and then verifying with a shared secret would be strictly
    weaker than what the operator asked for, which is the worst outcome available.
    """
    with pytest.raises(AuthConfigurationError, match="not supported"):
        TenantResolver(_settings(jwt_secret=SECRET, jwt_algorithm="RS256"))


# --- header handling -------------------------------------------------------


@pytest.fixture
def resolver() -> TenantResolver:
    return TenantResolver(_settings(jwt_secret=SECRET))


def test_valid_token_yields_the_claim_tenant(resolver: TenantResolver) -> None:
    principal = resolver.resolve(f"Bearer {_token()}")
    assert principal.tenant_id == "acme"
    assert principal.subject == "user-1"
    assert principal.mode == "jwt"


def test_missing_header_is_401_with_a_challenge(resolver: TenantResolver) -> None:
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(None)
    assert exc.value.status_code == 401
    # RFC 6750: without the challenge a client cannot tell it should authenticate.
    assert exc.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.parametrize(
    "header",
    ["", "Basic abc123", "Bearer", "Bearer    ", "bearertoken", _token()],
    ids=["empty", "wrong-scheme", "no-token", "blank-token", "no-space", "bare-token"],
)
def test_malformed_authorization_headers_are_401(resolver: TenantResolver, header: str) -> None:
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(header)
    assert exc.value.status_code == 401


def test_lowercase_bearer_scheme_is_accepted(resolver: TenantResolver) -> None:
    """The scheme is case-insensitive per RFC 7235; rejecting it would be a bug."""
    assert resolver.resolve(f"bearer {_token()}").tenant_id == "acme"


# --- signature and claim verification --------------------------------------


def test_token_signed_with_another_key_is_rejected(resolver: TenantResolver) -> None:
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(f"Bearer {_token(secret='another-signing-secret-32-bytes!!')}")
    assert exc.value.status_code == 401
    # The message must not distinguish "bad signature" from "malformed": that tells an
    # attacker which half of the token to keep working on.
    assert exc.value.detail == "token is not valid"


def test_unsigned_token_is_rejected(resolver: TenantResolver) -> None:
    """
    `alg: none` is the classic JWT bypass: a token with no signature that a naive
    verifier accepts because it trusts the token's own algorithm header.
    """
    unsigned = jwt.encode(
        {"sub": "attacker", "tenant_id": "victim", "exp": int(time.time()) + 300},
        key="",
        algorithm="none",
    )
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(f"Bearer {unsigned}")
    assert exc.value.status_code == 401


def test_expired_token_is_rejected(resolver: TenantResolver) -> None:
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(f"Bearer {_token(exp=int(time.time()) - 1)}")
    assert exc.value.status_code == 401
    assert exc.value.detail == "token has expired"


def test_token_without_an_expiry_is_rejected(resolver: TenantResolver) -> None:
    """A token with no `exp` is a permanent credential. Refuse it rather than honour it."""
    forever = jwt.encode({"sub": "u", "tenant_id": "acme"}, SECRET, algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(f"Bearer {forever}")
    assert exc.value.status_code == 401
    assert "exp" in exc.value.detail


def test_missing_tenant_claim_is_403_not_401(resolver: TenantResolver) -> None:
    """
    The credential is valid; it just authorises nothing. 401 would tell the client to
    retry with a credential, and retrying with this one will never work.
    """
    token = jwt.encode({"sub": "u", "exp": int(time.time()) + 300}, SECRET, algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(f"Bearer {token}")
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    "value",
    ["", "  ", "../other", "tenant with spaces", "a" * 65, "-leading-dash", 42, None, ["acme"]],
    ids=[
        "empty",
        "blank",
        "traversal",
        "spaces",
        "too-long",
        "leading-dash",
        "int",
        "null",
        "list",
    ],
)
def test_malformed_tenant_claims_are_403(resolver: TenantResolver, value: object) -> None:
    """
    A claim that is not a plausible tenant id must not reach the RLS setting.

    Parameterised queries make this not an injection defence. It is here so a
    malformed claim fails loudly instead of resolving to some tenant that exists but is
    not the caller's -- or to a GUC value that matches no rows and looks like an empty
    corpus.
    """
    token = jwt.encode(
        {"sub": "u", "tenant_id": value, "exp": int(time.time()) + 300},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(f"Bearer {token}")
    assert exc.value.status_code == 403


def test_custom_tenant_claim_name_is_honoured() -> None:
    resolver = TenantResolver(_settings(jwt_secret=SECRET, jwt_tenant_claim="org"))
    token = jwt.encode(
        {"sub": "u", "org": "beta-corp", "tenant_id": "ignored", "exp": int(time.time()) + 300},
        SECRET,
        algorithm="HS256",
    )
    assert resolver.resolve(f"Bearer {token}").tenant_id == "beta-corp"


# --- audience and issuer ---------------------------------------------------


def test_audience_mismatch_is_rejected() -> None:
    resolver = TenantResolver(_settings(jwt_secret=SECRET, jwt_audience="report-api"))
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(f"Bearer {_token(aud='some-other-api')}")
    assert exc.value.status_code == 401


def test_audience_is_accepted_when_it_matches() -> None:
    resolver = TenantResolver(_settings(jwt_secret=SECRET, jwt_audience="report-api"))
    assert resolver.resolve(f"Bearer {_token(aud='report-api')}").tenant_id == "acme"


def test_issuer_mismatch_is_rejected() -> None:
    resolver = TenantResolver(_settings(jwt_secret=SECRET, jwt_issuer="https://idp.example"))
    with pytest.raises(HTTPException) as exc:
        resolver.resolve(f"Bearer {_token(iss='https://evil.example')}")
    assert exc.value.status_code == 401


def test_a_token_carrying_an_audience_is_fine_when_none_is_configured(
    resolver: TenantResolver,
) -> None:
    """
    PyJWT rejects an `aud` claim when the caller expects no audience, so verification
    has to be switched off explicitly rather than left at its default. Without this,
    every token from an IdP that sets `aud` would fail for a service that does not
    configure one.
    """
    assert resolver.resolve(f"Bearer {_token(aud='anything')}").tenant_id == "acme"


def test_a_short_signing_secret_is_rejected_at_construction() -> None:
    """
    RFC 7518 section 3.2 requires an HMAC key at least as long as the hash output.

    PyJWT only warns. A 16-byte secret verifies tokens perfectly well until someone
    brute-forces it offline from a single captured token, at which point they can mint
    a token for any tenant -- which makes every other check in this module decorative.
    """
    with pytest.raises(AuthConfigurationError, match="too short"):
        TenantResolver(_settings(jwt_secret="sixteen-bytes-ok"))


def test_longer_algorithms_demand_longer_secrets() -> None:
    """HS512 needs 64 bytes; a 32-byte secret is fine for HS256 and not for HS512."""
    thirty_two = "unit-test-signing-secret-32bytes!!"
    TenantResolver(_settings(jwt_secret=thirty_two, jwt_algorithm="HS256"))
    with pytest.raises(AuthConfigurationError, match="minimum 64"):
        TenantResolver(_settings(jwt_secret=thirty_two, jwt_algorithm="HS512"))
