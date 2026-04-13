# GitHub Copilot Model Registry Changes for litellm

## Changes to `model_prices_and_context_window_backup.json`

### 1. New model entries (7 models)

Added missing entries for models available on the GitHub Copilot API.

| Key | max_input | max_output | mode | web_search | supported_endpoints |
|---|---|---|---|---|---|
| `github_copilot/gpt-5.4` | 272000 | 128000 | responses | true | cc, responses |
| `github_copilot/gpt-5.4-mini` | 272000 | 128000 | responses | true | responses |
| `github_copilot/gpt-5.2-codex` | 272000 | 128000 | responses | true | responses |
| `github_copilot/claude-sonnet-4.6` | 200000 | 128000 | chat | - | cc |
| `github_copilot/claude-opus-4.6` | 200000 | 128000 | chat | - | cc |
| `github_copilot/claude-opus-4.7` | 128000 | 128000 | chat | - | cc |
| `github_copilot/claude-opus-4.8` | 128000 | 16000 | chat | - | cc |

### 2. Fixed max_output_tokens (4 models)

Existing entries had `max_output_tokens: 16000` which is incorrect. Verified by sending
`max_tokens: 999999` to the Copilot `/v1/messages` endpoint and reading the error message
which reports the actual limit.

| Key | Before | After | Verification |
|---|---|---|---|
| `github_copilot/claude-haiku-4.5` | 16000 | 200000 | "exceeds the model limit of 200000" |
| `github_copilot/claude-sonnet-4.5` | 16000 | 200000 | "exceeds the model limit of 200000" |
| `github_copilot/claude-opus-4.5` | 16000 | 64000 | "max_tokens: 999999 > 64000" |
| `github_copilot/claude-sonnet-4` | 16000 | 65536 | "exceeds the model limit of 65536" |

### 3. Added supported_endpoints (8 models)

These models were missing `supported_endpoints` in the JSON but are confirmed working
via HTTP 200 responses from the Copilot API.

| Key | Added endpoints |
|---|---|
| `github_copilot/gemini-2.5-pro` | `/v1/chat/completions` |
| `github_copilot/gpt-3.5-turbo` | `/v1/chat/completions` |
| `github_copilot/gpt-3.5-turbo-0613` | `/v1/chat/completions` |
| `github_copilot/gpt-4.1` | `/v1/chat/completions` |
| `github_copilot/gpt-4.1-2025-04-14` | `/v1/chat/completions` |
| `github_copilot/gpt-4o-mini` | `/v1/chat/completions` |
| `github_copilot/gpt-4o-mini-2024-07-18` | `/v1/chat/completions` |
| `github_copilot/gpt-5-mini` | `/v1/chat/completions`, `/v1/responses` |

### 4. Deprecated models (not modified, noted here)

These models return HTTP 400 "The requested model is not supported" on both endpoints.
They are kept in the registry for historical reference.

| Key | Status |
|---|---|
| `github_copilot/claude-opus-4.6-fast` | Deprecated (400 on both endpoints) |
| `github_copilot/claude-opus-41` | Deprecated (400 on both endpoints) |
| `github_copilot/gpt-5` | Deprecated (400 on both endpoints) |
| `github_copilot/gpt-5.1-codex-max` | Deprecated (400 on both endpoints) |
| `github_copilot/gpt-4` | Deprecated (400 on both endpoints) |
| `github_copilot/gpt-4-0613` | Deprecated (400 on both endpoints) |
| `github_copilot/gpt-4-o-preview` | Deprecated (400 on both endpoints) |
| `github_copilot/gpt-4o` | Deprecated (400 on both endpoints) |
| `github_copilot/gpt-4o-2024-05-13` | Deprecated (400 on both endpoints) |
| `github_copilot/gpt-4o-2024-08-06` | Deprecated (400 on both endpoints) |
| `github_copilot/gpt-4o-2024-11-20` | Deprecated (400 on both endpoints) |

---

## Verification Method

All endpoint support was verified by sending minimal requests to the GitHub Copilot API
(`https://api.githubcopilot.com`) with proper authentication headers:

```
Copilot-Integration-Id: copilot-developer-cli
User-Agent: GitHubCopilotCLI/1.0.4
Editor-Version: copilot-cli/1.0.4
Editor-Plugin-Version: copilot-cli/1.0.4
```

### Endpoint test payloads

**Chat Completions** (`POST /chat/completions`):
```json
{"model":"MODEL","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":16}
```

**Responses API** (`POST /responses`):
```json
{"model":"MODEL","input":"hi","max_output_tokens":16}
```

### Full endpoint test results (2026-04-11)

```
Model                                    cc     resp   cc_detail                                resp_detail
============================================================================================================
claude-haiku-4.5                         200    400    ok                                       does not support Responses API
claude-opus-4.5                          200    400    ok                                       does not support Responses API
claude-opus-4.6                          200    400    ok                                       does not support Responses API
claude-opus-4.6-fast                     400    400    The requested model is not supported      The requested model is not supported
claude-opus-41                           400    400    The requested model is not supported      The requested model is not supported
claude-sonnet-4                          200    400    ok                                       does not support Responses API
claude-sonnet-4.5                        200    400    ok                                       does not support Responses API
claude-sonnet-4.6                        200    400    ok                                       does not support Responses API
gemini-2.5-pro                           200    400    ok                                       is not supported via Responses API
gemini-3-pro-preview                     400    400    The requested model is not supported      The requested model is not supported
gpt-3.5-turbo                            200    400    ok                                       is not supported via Responses API
gpt-3.5-turbo-0613                       200    400    ok                                       is not supported via Responses API
gpt-4                                    400    400    The requested model is not supported      The requested model is not supported
gpt-4-0613                               400    400    The requested model is not supported      The requested model is not supported
gpt-4-o-preview                          400    400    The requested model is not supported      The requested model is not supported
gpt-4.1                                  200    400    ok                                       is not supported via Responses API
gpt-4.1-2025-04-14                       200    400    ok                                       is not supported via Responses API
gpt-4o                                   400    400    The requested model is not supported      The requested model is not supported
gpt-4o-2024-05-13                        400    400    The requested model is not supported      The requested model is not supported
gpt-4o-2024-08-06                        400    400    The requested model is not supported      The requested model is not supported
gpt-4o-2024-11-20                        400    400    The requested model is not supported      The requested model is not supported
gpt-4o-mini                              200    400    ok                                       is not supported via Responses API
gpt-4o-mini-2024-07-18                   200    400    ok                                       is not supported via Responses API
gpt-5                                    400    400    The requested model is not supported      The requested model is not supported
gpt-5-mini                               200    200    ok                                       ok
gpt-5.1                                  200    200    ok                                       ok
gpt-5.1-codex-max                        400    400    The requested model is not supported      The requested model is not supported
gpt-5.2                                  200    200    ok                                       ok
gpt-5.2-codex                            400    200    not accessible via /chat/completions      ok
gpt-5.3-codex                            400    200    not accessible via /chat/completions      ok
gpt-5.4                                  200    200    ok                                       ok
gpt-5.4-mini                             400    200    not accessible via /chat/completions      ok
claude-opus-4.8                          200    400    ok                                       does not support Responses API
```

### Web search test results (2026-04-11)

Tested by sending web search requests and checking raw response for actual search execution.

**Chat Completions** with `"web_search_options": {}`:
All models that accept chat/completions returned HTTP 200 but **silently ignored** the
web search parameter — responses contained "I don't have live web access" type messages
with no annotations or citations.

**Responses API** with `"tools": [{"type": "web_search"}]`:

| Model | Status | Web search calls | Result |
|---|---|---|---|
| gpt-5.4 | 200 | 3+ search calls | Real search results with url_citation annotations |
| gpt-5.4-mini | 200 | 3+ search calls | Real search results with url_citation annotations |
| gpt-5.3-codex | 200 | search + open_page | Real search results |
| gpt-5.2-codex | 200 | 4 search calls | Real search results |
| gpt-5.2 | 200 | 2 search calls | Real search results |
| gpt-5.1 | 401 | - | Unauthorized |
| gpt-5-mini | 401 | - | Unauthorized |
| gpt-4.1 | 401 | - | Not supported via Responses API |

**Conclusion**: Web search only works via Responses API `tools:[{"type":"web_search"}]`.
The `web_search_options` parameter on `/chat/completions` is silently ignored by the
GitHub Copilot API for all models.

### reasoning_effort validation (2026-06-01)

Tested `reasoning_effort` on Copilot chat/completions using the development
Copilot token directory. Both Claude models accepted all tested effort values:

| Model | minimal | low | medium | high | xhigh |
|---|---:|---:|---:|---:|---:|
| `claude-opus-4.7` | 200 | 200 | 200 | 200 | 200 |
| `claude-opus-4.8` | 200 | 200 | 200 | 200 | 200 |

These flags are specific to the `github_copilot` provider. Native Anthropic
Claude entries intentionally omit `supports_minimal_reasoning_effort`; LiteLLM
maps `minimal` to Anthropic `output_config.effort=low` for those providers.

### max_output_tokens verification (2026-04-11)

Verified by sending `max_tokens: 999999` to `/v1/messages` and reading the error:

```
claude-sonnet-4.6: The maximum tokens you requested exceeds the model limit of 128000
claude-opus-4.6:   The maximum tokens you requested exceeds the model limit of 128000
claude-opus-4.5:   max_tokens: 999999 > 64000, which is the maximum allowed number of output tokens for claude-opus-4-5-20251101
claude-sonnet-4.5: The maximum tokens you requested exceeds the model limit of 200000
claude-haiku-4.5:  The maximum tokens you requested exceeds the model limit of 200000
claude-sonnet-4:   The maximum tokens you requested exceeds the model limit of 65536
```
