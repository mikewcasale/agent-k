Structured Agent Guidance Embeddings (SAGE)
==========================================

**A Documentation Standard for Human ↔ Agent Collaborative Codebases**

**Version:** 1.3.6\
**Date:** 2026-01-25\
**Status:** Release Candidate (Implementable)

* * * * *

Executive Summary
-----------------

SAGE defines a **hybrid documentation system** optimized for:

1.  **Coding agents** (Claude Code, Codex, Cursor, etc.) that must reliably navigate a codebase, reuse existing components, and adhere to architectural patterns without hallucinating or duplicating code.

2.  **Human reviewers** who need fast, high-signal context to validate agent changes and catch common failure modes (redundant abstractions, inconsistent DI, "creative" new modules, silent behavior changes).

SAGE synthesizes strengths from multiple docstring approaches:

-   **Minimalism (Uvicorn-style):** Prefer type hints and descriptive names; avoid drift-prone prose.

-   **Signature co-location (FastAPI-style):** Document parameters inline with `Annotated[..., Doc(...)]` so parameter docs don't drift.

-   **Narrative readability (Google/pydantic-ai-style):** Keep a familiar summary + optional examples for humans.

-   **Dynamic context discovery (Cursor-style):** Extract metadata into `.sage/` artifacts agents can grep/query on-demand.

**Core innovation:** SAGE introduces a machine-parseable **Contextual Relationship Graph (CRG)** embedded in docstrings using `@` tags. These tags encode:

-   Canonical ownership ("where does this live?")

-   Similarity/disambiguation ("don't reinvent this")

-   Safe extension points vs sealed internals

-   Dependency injection and lifecycle expectations

-   Anti-pattern warnings tuned to agent failure modes

**NatSpec-inspired compatibility (Solidity):** SAGE supports audience-separated narrative (`@notice` vs `@dev`), inheritance reuse (`@inheritdoc`), and structured custom metadata (`@custom:<name>`), adapted to Python and to agent workflows.

**Optional automation:** SAGE supports an optional **knowledge-graph MCP server** that can query and propose doc updates from extracted CRG + static analysis, while enforcing **human approval** for non-mechanical edits.

### Release Candidate Notes (v1.3.6 vs v1.3.5)

This release candidate is a cleanup for public release:

-   **Fixed tag grammar** to explicitly support `@custom:<name>` without contradicting the base tag grammar.

-   **Removed "hybrid" tag forms** (scalar + continuation) by defining block-map patterns for tags that need subfields (e.g., `@pure`, `@total`, `@factory-for`).

-   **Made multi-line `@notice`/`@dev` implementable** via YAML block scalars (`@notice: |`).

-   **Standardized booleans** to `true`/`false` everywhere in structured blocks and recommended normalizing enums to lowercase.

-   **Aligned extraction examples with schemas** (consistent use of root `id`, consistent key names).

-   **Adjusted validation semantics** to avoid false contradictions (e.g., total functions may still raise invalid-input errors).

* * * * *

Table of Contents
-----------------

1.  Design Goals

2.  Normative Language

3.  Core Concepts

4.  Docstring Layers and Audiences

5.  Tag Syntax, Parsing, and Normalization

6.  Component Reference Syntax

7.  Parameter Documentation

8.  Semantic Tags

9.  Module-Level Documentation

10. Class-Level Documentation

11. Method-Level Documentation

12. Attribute-Level Documentation

13. Function-Level Documentation

14. Private Member Documentation

15. Inheritance and Cross-References

16. Custom Tags

17. Dynamic Context Discovery

18. Machine-Readable Extraction Model

19. Knowledge-Graph MCP Integration

20. Agent Integration Protocol

21. Human Reviewer Support

22. Validation Rules

23. Adoption Strategy

24. Complete Templates

25. Quick Reference

26. Appendices

* * * * *

1\. Design Goals
----------------

### 1.1 Primary Goals (MUST)

| Goal | Description |
| --- | --- |
| Prevent agent duplication | Agents MUST discover canonical homes and "already exists" alternatives via `@graph`, `@canonical-home`, and `@similar`. |
| Make architectural intent explicit | The "why" and "why not" MUST be discoverable via `@pattern`, `@prefer-over`, and `@anti-patterns`. |
| Make constraints machine-readable | Constraints MUST be extractable via deterministic parsing rules (no NLP required). |
| Minimize documentation drift | Parameter docs MUST be co-located with signatures when possible; structured tags SHOULD be mechanically verifiable. |
| Support dual audiences | Documentation MUST serve both humans and agents with appropriate detail levels. |

### 1.2 Secondary Goals (SHOULD)

| Goal | Description |
| --- | --- |
| Human-friendly | SHOULD remain readable and compatible with typical doc generators (Sphinx, MkDocs, griffe). |
| Brownfield-friendly | SHOULD support incremental adoption in existing codebases. |
| Minimalism-compatible | SHOULD work even when many components have minimal docstrings. |
| Tooling-enabled | SHOULD support extraction, validation, and optional MCP-backed querying. |

### 1.3 Non-Goals

-   SAGE is not a replacement for architecture docs (ADRs, design docs).

-   SAGE does not mandate runtime enforcement.

-   SAGE does not require long docstrings everywhere.

-   SAGE does not mandate a specific doc generator.

* * * * *

2\. Normative Language
----------------------

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are used as defined in RFC 2119.

* * * * *

3\. Core Concepts
-----------------

### 3.1 Component

A **component** is any documentable code entity:

-   Package or module

-   Class (including dataclasses, Pydantic models, Protocols)

-   Function or method

-   Attribute or field

-   Type alias or constant

### 3.2 Canonical Home

For most non-trivial concepts, there is a **canonical home**---the authoritative location where the concept is implemented or extended.

**Agent behavior (MUST):** before creating new artifacts, agents MUST search canonical homes using:

-   `.sage/canonical-homes.json` (if present), and/or

-   `@canonical-home` / `@graph.provides`, and/or

-   `@similar` entries in relevant modules/classes.

### 3.3 Contextual Relationship Graph (CRG)

The CRG is SAGE's embedded "knowledge graph":

| Element | Description |
| --- | --- |
| Nodes | Components (modules/classes/functions/attributes) |
| Edges | Relationships: `provides`, `consumes`, `similar`, `replaces`, `calls`, `extends`, `implements` |
| Policies | Constraints: `@agent-guidance`, `@anti-patterns`, `@extends.sealed`, DI patterns |

The CRG can be:

1.  Embedded in docstrings (primary)

2.  Extracted to `.sage/` files (recommended)

3.  Served via MCP (optional)

* * * * *

4\. Docstring Layers and Audiences
----------------------------------

### 4.1 Narrative Layer (Human-first)

When a docstring exists, it MUST start with a **one-line summary**. Additional prose is optional.

Recommended narrative elements (as needed):

-   Extended description

-   Example(s) for public APIs

-   "Returns" / "Raises" narrative sections when useful to humans

### 4.2 Structured Layer (Machine-first)

Structured metadata uses `@` tags designed for deterministic parsing. This is the primary layer for agents and tooling.

### 4.3 NatSpec-Inspired Audience Split (OPTIONAL but recommended for public APIs)

| Tag | Audience | Purpose |
| --- | --- | --- |
| `@notice` | End-users, reviewers | "Safe to publish" description (API docs, tooltips). |
| `@dev` | Developers, agents | Implementation context: invariants, caveats, tradeoffs, extension guidance. |

**Rule:** If `@notice` exists, it SHOULD be consistent with the summary line. If they differ, `@notice` is authoritative for user-facing generated docs.

* * * * *

5\. Tag Syntax, Parsing, and Normalization
------------------------------------------

This section is intentionally strict to keep tooling implementable without heuristics.

### 5.1 Docstring Cleaning and Indentation Model

Parsers MUST parse the docstring content after **PEP 257-style cleaning** (i.e., equivalent to `inspect.getdoc()` dedentation).

All indentation rules in this spec apply **after** dedent.

### 5.2 Tag Header Grammar

A tag header MUST:

-   Begin at the start of a line (column 0 after dedent)

-   Start with `@`

-   Use a valid tag name

-   End with a colon `:`

#### 5.2.1 Standard Tags

Standard tag names MUST match:

`@[a-z][a-z0-9-]*(\.[a-z][a-z0-9-]*)*:`

Examples:

-   `@graph:`

-   `@agent-guidance:`

-   `@human-review:`

-   `@thread-safety:`

#### 5.2.2 Custom Tags (NatSpec-inspired)

Custom tags MUST use:

`@custom:<name>:`

Where `<name>` matches:

`[a-z][a-z0-9-]*(\.[a-z][a-z0-9-]*)*`

Examples:

-   `@custom:api-version:`

-   `@custom:stability:`

-   `@custom:feature-flag:`

-   `@custom:docs.sla-tier:`

> Note: `@custom:<name>` is treated as a distinct tag namespace; tooling extracts it under `custom`.

### 5.3 Two Allowed Value Forms (NO HYBRID FORM)

SAGE allows exactly two forms for any tag:

#### A) Scalar Form (single line)

`@idempotency: safe
@sealed: true`

Rules:

-   Scalar tags MUST NOT have indented continuation.

-   The entire value is on the same line.

-   Values are trimmed; empty scalar values are invalid.

#### B) Block Form (multi-line)

Block tags are either:

1.  A YAML-like block map/list:

`@graph:
    id: services.http_client
    provides:
        - services.http_client:HTTPClient`

1.  A YAML block scalar for multi-line text:

`@notice: |
    Canonical async HTTP client for outbound service calls.
    Use create_http_client() or DI; do not instantiate directly.`

Rules:

-   Block content MUST be indented **4 spaces** per level.

-   Tabs MUST NOT be used.

-   The tag header line MAY be `@tag:` or `@tag: |` or `@tag: >`.

-   **Hybrid forms are forbidden** (e.g., `@pure: true` plus indented subfields).

### 5.4 YAML-like Subset (Tool-friendly)

Within block tags, allowed structures are:

-   Maps (`key: value`)

-   Lists (`- item`)

-   Scalars (string/number/boolean)

-   Block scalars (`|` literal, `>` folded)

Boolean literals MUST be `true` / `false` (lowercase).

### 5.5 Quoting and Ambiguity

-   Values containing `:` followed by whitespace SHOULD be quoted.

-   URLs MUST be quoted.

-   Component references (e.g., `services.http_client:HTTPClient`) generally do **not** need quotes because the colon is not followed by whitespace.

Examples:

`url: "https://example.com"
reason: "Uses http: protocol semantics"`

### 5.6 Parsing Boundaries and Code Fences

Parsers MUST ignore tag headers that appear inside fenced code blocks:

-   A fenced block starts with a line beginning with ``` (triple backticks)

-   It ends with a line beginning with ``` (triple backticks)

This prevents examples from being misinterpreted as tags.

### 5.7 Duplicate Tags

If the same tag appears multiple times in a single docstring:

| Tag Type | Merge Behavior |
| --- | --- |
| Block-form maps | Merge keys in-order; later keys override earlier keys |
| Block-form lists | Concatenate in-order |
| Scalar tags | Last value wins |

If a tag is repeated with **type mismatch** (e.g., map then list), tools MUST emit an ERROR.

### 5.8 Key Normalization

For extraction outputs (JSON/YAML), tooling MUST normalize:

-   Kebab-case keys → snake_case (`anti-patterns` → `anti_patterns`)

-   Kebab-case tag names → snake_case at top-level (`agent-guidance` → `agent_guidance`)

-   Dots remain dots (namespacing), but each segment is normalized.

-   Custom tag keys after `custom:` are normalized and stored under `custom`.

Normalization function:

`def normalize_segment(seg: str) -> str:
    return seg.replace("-", "_")`

* * * * *

6\. Component Reference Syntax
------------------------------

SAGE is only effective if references are precise.

### 6.1 Canonical Component ID

A component reference SHOULD be one of:

| Format | Use Case | Example |
| --- | --- | --- |
| `module.path` | Module/package | `services.http_client` |
| `module.path:QualName` | Class/function/type alias | `services.http_client:HTTPClient` |
| `module.path:QualName.member` | Method/attribute | `services.http_client:HTTPClient.fetch` |

### 6.2 External References

For third-party libs, you MAY use either:

-   Canonical form if known: `aiohttp.client:ClientSession.get`

-   Dotted convenience form: `aiohttp.ClientSession.get`

**Validation policy:** tooling SHOULD validate internal references, but MAY treat external references as best-effort (WARN only).

### 6.3 Cross-Reference Link Syntax

Use:

`[DisplayName][module.path:QualName]`

Example:

`"""
See Also:
    - [HTTPClient][services.http_client:HTTPClient]
    - [fetch][services.http_client:HTTPClient.fetch]
"""`

* * * * *

7\. Parameter Documentation
---------------------------

### 7.1 Primary Rule: Co-locate in the Signature

Parameters SHOULD be documented using `Annotated[..., Doc(...)]` (FastAPI pattern) to avoid drift.

`from typing import Annotated
from sage import Doc, Constraint, Range, Default

async def fetch( path: Annotated[ str,
        Doc("URL path relative to base_url. MUST start with '/'."),
        Constraint(r"^/.*"),
    ],
    *,
    timeout: Annotated[
        float | None,
        Doc("Override timeout seconds. None uses instance default."),
        Range(0.0, 300.0),
        Default("self.default_timeout"),
    ] = None,
) -> Response:
    ...`

> Note: `Doc`, `Constraint`, `Range`, and `Default` are reference metadata helpers; projects MAY substitute equivalents. Tools SHOULD be configurable to recognize the project's chosen metadata classes.

### 7.2 Allowed Fallback: `@params` for Non-Annotatable Inputs

If `Annotated` is not cleanly usable (e.g., `**kwargs`, variadic patterns), you MAY use `@params`:

`"""Execute request with additional options.

@params:
    kwargs.timeout: "Timeout override in seconds."
    kwargs.retry: "Enable retry; disable for non-idempotent operations."
    args[0]: "First positional arg meaning ..."
"""`

Rules:

-   `@params` MUST NOT duplicate parameters already documented in `Annotated`.

-   Use dot notation for kwargs: `kwargs.<name>`

-   Use indexed notation for variadic: `args[0]`, `args[1]`

* * * * *

8\. Semantic Tags
-----------------

### 8.1 `@pure`

A function is **pure** if it has no observable side effects and is referentially transparent.

Allowed forms:

`@pure: true`

or:

`@pure:
    value: true
    reason: "Deterministic; no I/O or mutation."`

### 8.2 `@total`

A function is **total** if it produces an output for all **valid inputs**, and terminates.

Allowed forms:

`@total: true`

or:

`@total:
    value: true
    notes: "Raises ValueError only for invalid input."`

### 8.3 `@effects`

Use `@effects` to categorize side effects beyond simple state mutation:

`@effects:
    state:
        - order.status
        - metrics.order_count
    io:
        - database write
        - audit log
    external:
        - payment gateway
    environment:
        - "Sets ORDER_PROCESSING_LOCK during execution"`

**Recommendation:**

-   Use `@state-changes` for simple internal mutations.

-   Add `@effects` when there is I/O, external calls, or environment changes.

* * * * *

9\. Module-Level Documentation
------------------------------

### 9.1 When Required

Module docstrings are REQUIRED for modules that:

-   Export core public APIs, OR

-   Are canonical homes for major concepts, OR

-   Are agent-frequent entrypoints (agents commonly modify/extend).

### 9.2 Required Structure

If present, a module docstring MUST include:

1.  One-line summary

2.  `@graph` block for non-trivial modules

It SHOULD include:

-   `@similar` for disambiguation

-   `@agent-guidance` constraints

-   `@notice` and `@dev` for dual-audience output

### 9.3 Example

`"""HTTP client implementations for external service communication.

This module is the canonical home for outbound HTTP behavior (retry, circuit breaking,
instrumentation). External API calls MUST use HTTPClient from here.

@notice: |
    Canonical async HTTP client for outbound service calls.
    Use create_http_client() or dependency injection; do not instantiate directly.

@dev: |
    Repository-adapter pattern. All HTTP clients MUST be obtained through DI/factory.
    Agents MUST NOT create parallel HTTP implementations.

@graph:
    id: services.http_client
    provides:
        - services.http_client:HTTPClient
        - services.http_client:RetryPolicy
        - services.http_client:create_http_client
    consumes:
        - config.http:HTTPClientConfig
        - observability:Logger
    pattern: repository-adapter
    version: "2.1.0"

@similar:
    - id: services.grpc_client
      when: "Use for gRPC; this module is HTTP-only."
    - id: legacy.old_http
      when: "Deprecated since 2.0; migrate here."

@replaces:
    - module: legacy.old_http
      since: "2.0.0"

@agent-guidance:
    do:
        - "Search @similar targets before creating new HTTP-related modules/classes."
        - "Use DI/factory to obtain HTTPClient instances."
    do_not:
        - "Create new HTTP client classes. Extend/compose HTTPClient instead."
        - "Implement retry logic outside RetryPolicy."
        - "Use requests/urllib directly in production code."

@human-review:
    last-verified: 2026-01-15
    test-coverage: 94
    owners:
        - platform-team

@custom:api-version: "2.1"
@custom:stability: stable
"""`

* * * * *

10\. Class-Level Documentation
------------------------------

### 10.1 Significant Class Definition

A class is **significant** if it is:

-   Public API, OR

-   Instantiated by DI container/factory, OR

-   Used across modules, OR

-   A canonical home for a pattern/abstraction

### 10.2 Required Structure (for Significant Classes)

If a significant class has a docstring:

-   MUST include `@pattern`

-   MUST include `@collaborators` if it expects injected dependencies

-   SHOULD include `@extends` if extension points exist

-   SHOULD include `@anti-patterns` if it's easy to misuse

### 10.3 Example

`class HTTPClient:
    """Async HTTP client with retry and circuit breaking.

    @notice: |
        Canonical async HTTP client for outbound service calls.
        Obtain instances through DI or create_http_client().

    @dev: |
        Prefer composition for auth/serialization. Subclassing is tightly scoped.
        Internal state is sealed and MUST NOT be accessed.

    @pattern:
        name: singleton-per-config
        rationale: "Connection pooling requires a stable instance per base URL."
        violations: "Multiple instances can cause socket exhaustion under load."
        enforcement: "Factory validates; __init__ is not used directly."

    @extends:
        allowed:
            - services.http_client.extensions:CustomAuth
            - services.http_client.extensions:CustomSerializer
        sealed:
            - services.http_client._pool:_ConnectionPool
            - services.http_client._retry:_RetryState
        guidance: "Extend via composition; subclass only for auth/serialization hooks."
        hooks:
            - on_request
            - on_response
            - on_error

    @implements:
        - protocols.http:HTTPClientProtocol

    @collaborators:
        required:
            - observability:Logger
            - config.http:HTTPClientConfig
        optional:
            - observability:MetricsCollector
        injection: constructor
        lifecycle: "Managed by DI container; one instance per base_url."

    @invariants:
        - "base_url is immutable after construction."
        - "timeout > 0 (0 means no-wait, not infinite)."
        - "max_retries in [0, 10]."

    @anti-patterns:
        - description: "Creating HTTPClient per-request"
          instead: "Use DI/container; instantiation is expensive and breaks pooling."
        - description: "Subclassing to add headers"
          instead: "Use middleware hooks or header factory."

    @concurrency:
        level: conditional
        model: coroutines
        notes: "Safe for concurrent coroutines; not safe across OS threads."
    """`

> Note: v1.3.6 introduces `@concurrency` as the clearer name for what v1.3.5 called `@thread-safety`. Tooling SHOULD accept `@thread-safety` as an alias, but new docs SHOULD use `@concurrency`.

* * * * *

11\. Method-Level Documentation
-------------------------------

Method docstrings SHOULD focus on behavior and agent navigation; parameters are documented in `Annotated`.

### 11.1 Example

`async def fetch( self,
    path: Annotated[str, Doc("Path relative to base_url; MUST start with '/'.")],
    *,
    timeout: Annotated[float | None, Doc("Override timeout seconds.")] = None,
) -> Response:
    """Execute an HTTP GET with retry and circuit breaker protection.

    Example:
        ```python
        response = await client.fetch("/api/users", timeout=5.0)
        if response.ok:
            users = response.json()
        ```

    @notice: "Performs a GET request with automatic retry and circuit breaking."

    @dev: |
        Primary GET entrypoint. Updates circuit breaker state on each call.

    @pure:
        value: false
        reason: "Network I/O; updates circuit breaker state."

    @total:
        value: false
        notes: "Raises on invalid path/auth failures."

    @idempotency: safe

    @complexity:
        time: "O(1) amortized; O(n) worst-case where n=max_retries."
        space: "O(1) excluding response body."

    @state-changes:
        - "Updates circuit breaker state on failures."
        - "Records request metrics counters."

    @effects:
        state:
            - _circuit_breaker
            - _request_count
        io:
            - network request
            - connection pool management
        external:
            - target HTTP server

    @errors:
        recoverable:
            - TimeoutError
            - ConnectionError
        terminal:
            - AuthenticationError
            - ValueError
        notes: "Recoverable errors trigger retry; terminal errors propagate."

    @calls:
        internal:
            - services.http_client:HTTPClient._execute_with_retry
        external:
            - aiohttp.client:ClientSession.get

    @prefer-over:
        aiohttp.ClientSession.get: "Missing retry/circuit breaker integration."
        requests.get: "Blocking; do not use in async code."

    @human-review:
        edge-cases:
            - "Empty path should raise ValueError, not normalize."
            - "Timeout=0 means no-wait, not infinite."
        test-coverage: 96
    """`

* * * * *

12\. Attribute-Level Documentation
----------------------------------

### 12.1 Required Sensitivity

If an attribute has an attribute-level docstring, it MUST include `@sensitivity`.

Sensitive fields (PII/secret) SHOULD always have attribute docstrings.

### 12.2 Sensitivity Values

Canonical values (recommended lowercase):

-   `none`

-   `internal`

-   `pii`

-   `secret`

Tooling MAY accept `PII` and normalize to `pii`.

### 12.3 Example

`@dataclass
class RequestContext:
    """Immutable request metadata for tracing/auth."""

    request_id: str
    """Unique identifier for distributed tracing.

    @source: "Generated by TraceMiddleware; never user-provided."
    @validation: "UUID v4 lowercase with hyphens."
    @sensitivity: none
    @invariants:
        - "Always 36 chars."
        - "Immutable after creation."
    """

    user_id: str | None
    """Authenticated user ID; None for anonymous.

    @source: "JWT claim 'sub' via AuthMiddleware."
    @sensitivity: pii
    @logging: "MUST be masked or omitted in logs."
    """

    api_key_hash: str | None = None
    """Hashed API key for rate limiting.

    @source: "Derived from X-API-Key header via SHA-256."
    @sensitivity: secret
    @logging: "MUST NOT be logged."
    """`

* * * * *

13\. Function-Level Documentation
---------------------------------

### 13.1 Factories

Factory functions SHOULD declare what they create and why.

`def create_http_client( base_url: Annotated[str, Doc("Base URL. MUST include scheme.")],
    *,
    config: Annotated[Config | None, Doc("Config; None uses defaults.")] = None,
) -> HTTPClient:
    """Factory for HTTPClient with validated parameters.

    @notice: |
        Creates a properly configured HTTPClient.
        Preferred creation path outside DI.

    @dev: |
        Validates parameters and resolves defaults. Do not call HTTPClient.__init__.

    @factory-for:
        id: services.http_client:HTTPClient
        rationale: "Validation + caching too complex for __init__."
        singleton: true
        cache-key: base_url

    @canonical-home:
        for:
            - "http client construction"
        notes: "Canonical creation path for HTTPClient instances."

    @pure:
        value: false
        reason: "Creates/returns cached client instance."

    @total:
        value: true
        notes: "Raises ValueError for invalid base_url."

    @agent-guidance:
        do:
            - "Use this factory for HTTPClient instantiation outside DI."
        do_not:
            - "Call HTTPClient.__init__ directly."
            - "Create parallel factory functions."

    @errors:
        terminal:
            - ValueError
            - ConfigurationError
    """`

* * * * *

14\. Private Member Documentation
---------------------------------

Private members SHOULD be documented only when:

-   Non-obvious behavior exists, OR

-   There is security risk, OR

-   There is known technical debt, OR

-   It is a sealed internal used by agents incorrectly.

`def _execute_with_retry(self, request: PreparedRequest) -> Response:
    """Internal: execute request with retry policy applied.

    @internal: "Do not call directly; use public fetch/post/etc."
    @sealed: true

    @dev: |
        Core retry loop. Agents MUST NOT duplicate this logic elsewhere.

    @tested-via:
        - "tests/test_http_client.py::TestRetryBehavior"
    """`

* * * * *

15\. Inheritance and Cross-References
-------------------------------------

### 15.1 `@inheritdoc` (NatSpec-inspired)

`class HTTPClient(HTTPClientProtocol):
    """Concrete implementation.

    @implements:
        - protocols.http:HTTPClientProtocol
    """

    async def fetch(self, path: str, **kwargs) -> Response:
        """@inheritdoc: protocols.http:HTTPClientProtocol.fetch

        @dev: |
            Adds circuit breaker protection beyond protocol requirements.

        @state-changes:
            - "Updates circuit breaker state"
        """`

### 15.2 Inheritance Merge Rules

When `@inheritdoc` is used:

1.  Parent tags are inherited.

2.  Child tags override inherited tags of the same name (except `@dev`).

3.  `@dev` is merged by concatenation (parent then child).

4.  If parameter names differ, inheritance MUST NOT apply.

* * * * *

16\. Custom Tags
----------------

### 16.1 `@custom:<name>` (NatSpec-inspired)

`"""
@custom:api-version: "2.1"
@custom:stability: stable
@custom:deprecation-date: "2027-01-01"
@custom:feature-flag: enable_new_retry_logic
"""`

Tooling MUST extract these into a `custom` object in JSON output.

### 16.2 Custom Tag Registry (Recommended)

`# .sage/custom-tags.yaml
custom_tags:
  api-version:
    description: "API version this component belongs to"
    type: string

  stability:
    description: "Stability level"
    type: enum
    values: [experimental, beta, stable, deprecated]
    required: true`

* * * * *

17\. Dynamic Context Discovery
------------------------------

SAGE recommends extracting doc metadata into `.sage/` artifacts so agents load only what's needed:

`.sage/
├── index.json
├── graph.json
├── patterns.json
├── anti-patterns.json
├── agent-guidance.json
├── canonical-homes.json
├── components/
│   ├── services.http_client.json
│   └── services.http_client.HTTPClient.json
└── similar/
    └── http-clients.md`

* * * * *

18\. Machine-Readable Extraction Model
--------------------------------------

### 18.1 Canonical JSON Output (Normalized)

Example module extraction:

`{
  "id": "services.http_client",
  "type": "module",
  "summary": "HTTP client implementations for external service communication.",
  "notice": "Canonical async HTTP client for outbound service calls.",
  "dev": "Repository-adapter pattern. All clients MUST be obtained via DI/factory.",
  "graph": {
    "id": "services.http_client",
    "provides": [
      "services.http_client:HTTPClient",
      "services.http_client:RetryPolicy",
      "services.http_client:create_http_client"
    ],
    "consumes": [
      "config.http:HTTPClientConfig",
      "observability:Logger"
    ],
    "pattern": "repository-adapter",
    "version": "2.1.0"
  },
  "similar": [
    { "id": "services.grpc_client", "when": "Use for gRPC; this module is HTTP-only." }
  ],
  "agent_guidance": {
    "do": ["Use DI/factory to obtain HTTPClient instances."],
    "do_not": ["Create new HTTP client classes."]
  },
  "human_review": {
    "last_verified": "2026-01-15",
    "test_coverage": 94,
    "owners": ["platform-team"]
  },
  "custom": {
    "api_version": "2.1",
    "stability": "stable"
  }
}`

* * * * *

19\. Knowledge-Graph MCP Integration
------------------------------------

SAGE can be paired with an MCP server that loads `.sage/` and optionally enriches it with:

-   AST symbol table

-   Import graph

-   Best-effort call graph

-   Git history hints (optional)

### 19.1 Human-in-the-Loop Requirement

Tools MUST NOT silently rewrite docstrings. They MAY propose patches, with human approval required for non-mechanical changes.

### 19.2 `@automation` Tag

Projects MAY mark tool-managed fields:

`"""
@automation:
    managed-by: sage-kg
    allow-write:
        - graph
        - calls
        - tested-via
    require-approval:
        - agent-guidance
        - anti-patterns
"""`

* * * * *

20\. Agent Integration Protocol
-------------------------------

Before creating new code, agents MUST:

1.  Read relevant `@graph` blocks

2.  Check `@similar` for existing alternatives

3.  Respect sealed internals (`@sealed`, `@extends.sealed`)

4.  Follow DI/lifecycle (`@collaborators`)

5.  Avoid documented `@anti-patterns`

6.  Prefer canonical construction paths (`@factory-for`, `@canonical-home`)

* * * * *

21\. Human Reviewer Support
---------------------------

SAGE tooling SHOULD generate checklists from extracted metadata (anti-patterns, invariants, sensitivity/logging rules, sealed violations).

* * * * *

22\. Validation Rules
---------------------

### 22.1 Structural

| Rule | Level | Description |
| --- | --- | --- |
| V001 | ERROR | Non-trivial modules MUST have `@graph`. |
| V002 | ERROR | Significant class docstrings MUST include `@pattern`. |
| V003 | ERROR | Classes with injected deps MUST include `@collaborators`. |
| V004 | ERROR | Attribute docstrings MUST include `@sensitivity`. |
| V005 | WARN | Public APIs SHOULD include `@notice` and `@dev`. |

### 22.2 Consistency

| Rule | Level | Description |
| --- | --- | --- |
| C001 | ERROR | Sealed items MUST NOT be subclassed/called externally. |
| C002 | ERROR | Declared DI strategy MUST match actual code. |
| C003 | WARN | `@similar` references SHOULD resolve to known components. |
| C004 | WARN | If `@pure.value=false`, consider adding `@effects`. |

### 22.3 Semantic Checks

| Rule | Level | Description |
| --- | --- | --- |
| S001 | WARN | `@pure.value=true` but `@effects.io` or `@effects.external` non-empty. |
| S002 | INFO | `@total.value=true` and `@errors.terminal` non-empty: ensure exceptions are invalid-input only. |

* * * * *

23\. Adoption Strategy
----------------------

-   **Level 1 (Minimal):** add `@graph`, `@similar`, `@agent-guidance` to key modules.

-   **Level 2 (Standard):** add `@pattern`, `@collaborators`, `@anti-patterns` to significant classes; add `@sensitivity` to sensitive attributes.

-   **Level 3 (Full):** extraction + CI validation + MCP querying.

* * * * *

24\. Complete Templates
-----------------------

### 24.1 Module Template

`"""One-line summary.

@notice: |
    User-facing description.

@dev: |
    Developer/agent context.

@graph:
    id: module.path
    provides:
        - module.path:Thing
    consumes:
        - module.path:Dependency
    pattern: pattern-name

@similar:
    - id: other.module
      when: "Use when ..."

@agent-guidance:
    do:
        - "..."
    do_not:
        - "..."

@human-review:
    last-verified: YYYY-MM-DD
    test-coverage: NN
    owners:
        - team-name

@custom:stability: stable
"""`

### 24.2 Class Template

`class ClassName:
    """One-line summary.

    @notice: |
        User-facing description.

    @dev: |
        Developer notes.

    @pattern:
        name: pattern-name
        rationale: "Why this pattern"
        violations: "What breaks if violated"

    @extends:
        allowed:
            - module.path:ExtensionPoint
        sealed:
            - module.path:_InternalThing
        guidance: "How to extend"

    @collaborators:
        required:
            - module.path:RequiredDep
        optional:
            - module.path:OptionalDep
        injection: constructor
        lifecycle: "How managed"

    @anti-patterns:
        - description: "Mistake"
          instead: "Alternative"

    @concurrency:
        level: safe|unsafe|conditional
        model: threads|coroutines|both
        notes: "Details"
    """`

### 24.3 Method Template

`def method(self, x: Annotated[int, Doc("...")]) -> ReturnType:
    """One-line summary.

    @pure:
        value: false
        reason: "..."

    @total:
        value: true
        notes: "..."

    @idempotency: safe|unsafe

    @effects:
        io:
            - "..."

    @errors:
        recoverable: []
        terminal:
            - SomeError
    """`

### 24.4 Attribute Template

`field: Type
"""Brief description.

@source: "Where it comes from."
@sensitivity: none|internal|pii|secret
@logging: "Rules (required for pii/secret)."
"""`

* * * * *

25\. Quick Reference
--------------------

### 25.1 Core Tags

-   Navigation/graph: `@graph`, `@similar`, `@replaces`, `@canonical-home`

-   Architecture: `@pattern`, `@extends`, `@collaborators`, `@implements`

-   Agent constraints: `@agent-guidance`, `@anti-patterns`, `@prefer-over`

-   Semantics: `@pure`, `@total`, `@effects`, `@state-changes`, `@errors`

-   Audience: `@notice`, `@dev`

-   Inheritance: `@inheritdoc`

-   Automation: `@automation`

-   Custom: `@custom:<name>`

* * * * *

Appendices
----------

### Appendix A: JSON Schema (Representative)

> Note: This appendix is intentionally "representative" (not exhaustive). Projects commonly publish the full schema as separate versioned files under `.sage/schema/`.

#### Module Schema (v1.3.6)

`{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://sage.dev/schema/v1.3.6/module.json",
  "type": "object",
  "required": ["id", "type", "graph"],
  "properties": {
    "id": { "type": "string" },
    "type": { "const": "module" },
    "summary": { "type": "string" },
    "notice": { "type": "string" },
    "dev": { "type": "string" },
    "graph": {
      "type": "object",
      "required": ["id", "provides", "consumes"],
      "properties": {
        "id": { "type": "string" },
        "provides": { "type": "array", "items": { "type": "string" } },
        "consumes": { "type": "array", "items": { "type": "string" } },
        "pattern": { "type": "string" },
        "version": { "type": "string" }
      }
    },
    "custom": { "type": "object" }
  }
}`

### Appendix B: Minimal `Annotated` Metadata Helpers (Reference)

`from dataclasses import dataclass

@dataclass(frozen=True)
class Doc:
    text: str

@dataclass(frozen=True)
class Constraint:
    pattern: str

@dataclass(frozen=True)
class Range:
    min: float | int
    max: float | int

@dataclass(frozen=True)
class Default:
    expr: str`

* * * * *

Changelog
---------

### v1.3.6 (2026-01-25)

-   Fixed tag grammar to support `@custom:<name>` without contradicting base tag grammar

-   Enforced "two value forms" by removing hybrid scalar+block examples (`@pure`, `@total`, `@factory-for`, etc.)

-   Standardized multi-line `@notice` / `@dev` via `|` block scalars

-   Added `@concurrency` as clearer replacement for ambiguous `@thread-safety` (alias allowed)

-   Aligned JSON extraction examples with schema (`id`, normalized keys)

### v1.3.5 (2026-01-25)

-   Original v1.3.5 draft (pre-release)

* * * * *

License
-------

MIT License. (c) 2025--2026 SAGE Contributors