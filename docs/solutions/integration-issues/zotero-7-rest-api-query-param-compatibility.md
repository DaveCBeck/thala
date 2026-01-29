---
module: stores
date: 2025-12-16
problem_type: integration_issue
component: zotero
symptoms:
  - "GET/PATCH/DELETE requests to Zotero 7 items endpoint returning 405 or 400 errors"
  - "Query parameters ignored in Zotero 7 HTTP request handling"
  - "Plugin endpoints failing with unreliable behavior across GET/PATCH/DELETE methods"
  - "Zotero 7 plugin installation fails without update_url in manifest.json"
root_cause: wrong_api
resolution_type: code_fix
severity: high
tags: [zotero, zotero-7, api-migration, http-methods, query-parameters, plugin]
---

# Zotero 7 REST API Query Parameter Compatibility

## Problem

Zotero 7 changed how its HTTP server handles query parameters, breaking REST endpoints that used GET/PATCH/DELETE methods with query strings. The zotero-local-crud plugin's CRUD operations became unreliable when upgraded from Zotero 6 to Zotero 7.

### Environment

- **Module**: stores (core/stores/zotero.py)
- **Zotero**: 7.0.x (breaking change from 6.x)
- **Affected files**:
  - `core/stores/zotero.py`
  - `services/zotero/zotero-local-crud/bootstrap.js`
  - `services/zotero/zotero-local-crud/manifest.json`

### Symptoms

- GET, PATCH, DELETE operations on `/local-crud/item?key=XXX` return errors or silently fail
- Query parameter `key` not received by endpoint handler in Zotero 7
- Plugin installation fails with "applications.zotero.update_url not provided" error
- Inconsistent behavior: some requests work, others fail unpredictably

## Investigation

### What Didn't Work

1. **Using `request.query?.key` in Zotero 7**
   - Why it failed: Zotero 7 changed query parameter handling; `request.query` is unreliable or empty for GET/PATCH/DELETE methods

2. **Standard REST HTTP method routing (GET for read, PATCH for update, DELETE for remove)**
   - Why it failed: Zotero 7's HTTP server doesn't reliably route these methods with query parameters

## Root Cause

Zotero 7 changed its internal HTTP server implementation, breaking the standard REST pattern of using different HTTP methods with query parameters. The `request.query` object in endpoint handlers is unreliable for GET, PATCH, and DELETE methods.

```javascript
// BROKEN in Zotero 7 - query params unreliable
var ItemEndpoint = function() {};
ItemEndpoint.prototype = {
    supportedMethods: ["GET", "PATCH", "DELETE"],

    init: async function(request) {
        // request.query?.key is often undefined or empty in Zotero 7
        var key = request.query?.key;
    }
};
```

Additionally, Zotero 7 requires the `update_url` field in manifest.json for plugin installation to succeed.

## Solution

Change from multiple HTTP methods with query parameters to a single POST endpoint with an `action` field in the request body. This approach is more reliable because:
1. POST request bodies are always parsed correctly
2. Single endpoint simplifies routing
3. Action-based routing is explicit and debuggable

### Code Changes

**JavaScript (bootstrap.js) - Plugin endpoint:**

```javascript
// BEFORE (broken in Zotero 7)
var ItemEndpoint = function() {};
ItemEndpoint.prototype = {
    supportedMethods: ["GET", "PATCH", "DELETE"],

    init: async function(request) {
        var key = request.query?.key;  // Unreliable in Zotero 7

        switch (request.method) {
            case "GET": return this.handleGet(key);
            case "PATCH": return this.handleUpdate(key, request.data);
            case "DELETE": return this.handleDelete(key);
        }
    }
};

// AFTER (working in Zotero 7)
var ItemEndpoint = function() {};
ItemEndpoint.prototype = {
    supportedMethods: ["POST"],  // POST only

    init: async function(request) {
        var data = parseJSON(request.data);
        var key = data.key;              // Key from request body
        var action = data.action || "get";  // Action specifies operation

        switch (action) {
            case "get": return this.handleGet(item);
            case "update": return await this.handleUpdate(item, data);
            case "delete": return await this.handleDelete(item);
            default:
                return jsonResponse(400, { error: "Invalid action. Use: get, update, delete" });
        }
    }
};
```

**Python (core/stores/zotero.py) - Client wrapper:**

```python
# BEFORE (broken in Zotero 7)
async def get(self, zotero_key: str) -> Optional[ZoteroItem]:
    response = await client.get("/local-crud/item", params={"key": zotero_key})

async def update(self, zotero_key: str, updates: ZoteroItemUpdate) -> bool:
    response = await client.patch("/local-crud/item", params={"key": zotero_key}, json=payload)

async def delete(self, zotero_key: str) -> bool:
    response = await client.delete("/local-crud/item", params={"key": zotero_key})

# AFTER (working in Zotero 7)
async def get(self, zotero_key: str) -> Optional[ZoteroItem]:
    response = await client.post(
        "/local-crud/item",
        json={"action": "get", "key": zotero_key}
    )

async def update(self, zotero_key: str, updates: ZoteroItemUpdate) -> bool:
    payload = {"action": "update", "key": zotero_key}
    # ... add update fields ...
    response = await client.post("/local-crud/item", json=payload)

async def delete(self, zotero_key: str) -> bool:
    response = await client.post(
        "/local-crud/item",
        json={"action": "delete", "key": zotero_key}
    )
```

**manifest.json - Add required update_url:**

```json
{
  "applications": {
    "zotero": {
      "id": "zotero-local-crud@localhost",
      "update_url": "https://raw.githubusercontent.com/DaveCBeck/zotero-local-crud/main/updates.json",
      "strict_min_version": "7.0",
      "strict_max_version": "7.1.*"
    }
  }
}
```

### Files Modified

- `core/stores/zotero.py`: Changed GET/PATCH/DELETE to POST with action field
- `services/zotero/zotero-local-crud/bootstrap.js`: Changed ItemEndpoint to POST-only with action-based routing
- `services/zotero/zotero-local-crud/manifest.json`: Added update_url, updated version constraints
- `services/zotero/zotero-local-crud/README.md`: Updated API documentation
- `services/zotero/zotero-local-crud/notes.md`: Added development workflow documentation
- `services/zotero/custom-cont-init.d/20-install-plugins`: Added automatic plugin installation

## Prevention

### How to Avoid This

- **Test against target Zotero version**: Before deploying plugin updates, test CRUD operations against the specific Zotero version
- **Prefer POST with body over query params**: For plugin endpoints, POST with JSON body is more reliable than query parameters
- **Pin Zotero version constraints**: Use `strict_min_version` and `strict_max_version` in manifest.json
- **Monitor Zotero release notes**: Subscribe to Zotero release announcements for breaking changes
- **Add health check endpoint**: The `/local-crud/ping` endpoint validates plugin is loaded and working

### Test Case

```python
@pytest.mark.asyncio
async def test_zotero_post_action_endpoint():
    """Verify POST with action field works for all CRUD operations."""
    async with ZoteroStore() as store:
        # Create
        item = await store.create(ZoteroItemCreate(itemType="book", fields={"title": "Test"}))
        assert item.key

        # Get via POST
        fetched = await store.get(item.key)
        assert fetched.fields["title"] == "Test"

        # Update via POST
        success = await store.update(item.key, ZoteroItemUpdate(fields={"title": "Updated"}))
        assert success

        # Delete via POST
        deleted = await store.delete(item.key)
        assert deleted
```

## API Compatibility Matrix

| Zotero Version | Query Params (GET/PATCH/DELETE) | POST with Action | Status |
|----------------|--------------------------------|------------------|--------|
| 6.0-6.999 | Working | Not needed | EOL |
| 7.0-7.1 | Broken | Working | Supported |
| 8.0+ | Unknown | Unknown | Not tested |

## Related

- `services/zotero/zotero-local-crud/notes.md` - Development workflow and Docker setup
- `services/zotero/zotero-local-crud/README.md` - Complete API documentation
- `docs/architecture.md` - System architecture including Zotero service
