# Zotero Local CRUD Plugin - Development Notes

## Docker Development Workflow

### Fresh Start (Recommended for any code changes)
When editing `bootstrap.js` or `manifest.json`, always do a full restart:

```bash
docker compose exec zotero rm -rf /config/.zotero
docker compose restart
# Wait 60 seconds for Zotero to fully start
```

This is necessary because:
1. Zotero copies the XPI from `/opt/zotero/distribution/extensions/` to the profile's `extensions/` directory
2. Zotero caches addon code in `startupCache/` and `addonStartup.json.lz4`
3. Partial cache clearing often doesn't work - a full profile reset is cleanest

### How the Plugin Install Works
1. Init script `custom-cont-init.d/20-install-plugins` creates XPI from source files
2. XPI is placed in `/opt/zotero/distribution/extensions/`
3. On Zotero startup, it scans this directory and installs addons to the user profile
4. The `/opt/zotero/distribution` directory must be owned by `abc:abc` (the Zotero user)

### Key Files
- `manifest.json` - Plugin metadata (requires `update_url` field for Zotero 7)
- `bootstrap.js` - Plugin code with HTTP endpoint definitions
- `../custom-cont-init.d/20-install-plugins` - Init script that creates XPI

### Common Issues

#### "No endpoint found"
- Plugin not loaded. Check if XPI exists in distribution/extensions
- Try full restart with profile reset (see above)

#### NS_ERROR_FILE_ACCESS_DENIED
- Permission issue. Ensure `/opt/zotero/distribution` is owned by `abc:abc`
- The init script should handle this with `chown -R abc:abc /opt/zotero/distribution`

#### "applications.zotero.update_url not provided"
- Zotero 7 requires `update_url` in manifest.json's applications.zotero section

## API Design Notes

### Why POST-only for item operations?
Zotero 7 changed query parameter handling and it's unreliable. Instead of using
`GET /local-crud/item?key=XXX`, we use `POST /local-crud/item` with an `action`
field in the request body. This is more reliable and allows all operations through
a single endpoint.

### All Endpoints (all working)
```bash
# Health check
curl http://localhost:23119/local-crud/ping

# Create item
curl -X POST http://localhost:23119/local-crud/items \
  -H "Content-Type: application/json" \
  -d '{"itemType": "book", "fields": {"title": "Test"}}'

# Get item
curl -X POST http://localhost:23119/local-crud/item \
  -H "Content-Type: application/json" \
  -d '{"action": "get", "key": "XXXXXXXX"}'

# Update item
curl -X POST http://localhost:23119/local-crud/item \
  -H "Content-Type: application/json" \
  -d '{"action": "update", "key": "XXXXXXXX", "fields": {"title": "New Title"}}'

# Delete item
curl -X POST http://localhost:23119/local-crud/item \
  -H "Content-Type: application/json" \
  -d '{"action": "delete", "key": "XXXXXXXX"}'

# Search items
curl -X POST http://localhost:23119/local-crud/search \
  -H "Content-Type: application/json" \
  -d '{"conditions": [], "limit": 10}'
```
