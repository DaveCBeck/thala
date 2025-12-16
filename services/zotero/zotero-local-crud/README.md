# Zotero Local CRUD API

A Zotero 7 plugin that exposes HTTP REST endpoints for full CRUD operations on your local Zotero library without requiring cloud sync.

## Features

- **Full CRUD Operations**: Create, Read, Update, and Delete items via HTTP
- **Local Only**: No cloud sync or Zotero account required
- **REST-like API**: Simple JSON-based HTTP endpoints
- **Search Support**: Zotero's powerful search conditions via HTTP
- **No Authentication**: Designed for local automation (port 23119 is localhost-only)

## Installation

### From XPI File

1. Download the latest `zotero-local-crud.xpi` from releases
2. In Zotero, go to **Tools > Add-ons**
3. Click the gear icon and select **Install Add-on From File...**
4. Select the downloaded `.xpi` file

### From Source

```bash
cd zotero-local-crud
zip -r zotero-local-crud.xpi manifest.json bootstrap.js
```

Then install the generated `.xpi` file.

### Docker Setup

For running Zotero in Docker (using linuxserver/zotero image):

1. **Directory structure:**
```
services/zotero/
├── docker-compose.yml
├── custom-cont-init.d/
│   └── 20-install-plugins    # Init script to install plugin
├── zotero-local-crud/
│   ├── manifest.json
│   └── bootstrap.js
└── data/                     # Zotero config/data
```

2. **docker-compose.yml:**
```yaml
services:
  zotero:
    image: lscr.io/linuxserver/zotero:latest
    container_name: zotero
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=UTC
    volumes:
      - ./data:/config
      - ./custom-cont-init.d:/custom-cont-init.d:ro
      - ./zotero-local-crud:/opt/zotero-plugins/zotero-local-crud:ro
    ports:
      - "3001:3001"      # Web GUI
      - "23119:23120"    # Zotero HTTP API
```

3. **custom-cont-init.d/20-install-plugins:**
```bash
#!/usr/bin/with-contenv bash
# Install Zotero plugins as XPI files to distribution/extensions directory

PLUGIN_SOURCE="/opt/zotero-plugins/zotero-local-crud"
DIST_EXTENSIONS_DIR="/opt/zotero/distribution/extensions"

echo "**** Setting up Zotero plugins ****"

if ! command -v zip &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq zip >/dev/null 2>&1
fi

mkdir -p "$DIST_EXTENSIONS_DIR"

PLUGIN_ID="zotero-local-crud@localhost"
XPI_FILE="$DIST_EXTENSIONS_DIR/$PLUGIN_ID.xpi"

if [ -d "$PLUGIN_SOURCE" ]; then
    rm -f "$XPI_FILE"
    cd "$PLUGIN_SOURCE"
    zip -r "$XPI_FILE" manifest.json bootstrap.js
    chmod 644 "$XPI_FILE"
    chown -R abc:abc /opt/zotero/distribution
    echo "**** Created XPI at: $XPI_FILE ****"
else
    echo "**** Plugin source not found ****"
fi
```

4. **Start the container:**
```bash
docker compose up -d
# Wait ~60 seconds for Zotero to fully start
curl http://localhost:23119/local-crud/ping
```

### Development Workflow (Docker)

When editing the plugin code, you must restart with a clean profile:

```bash
docker compose exec zotero rm -rf /config/.zotero
docker compose restart
# Wait 60 seconds for full startup
```

See `notes.md` for detailed development notes.

### Git Subtree (for embedding in other repos)

```bash
# Add as subtree
git subtree add --prefix=path/to/zotero-local-crud \
  https://github.com/DaveCBeck/zotero-local-crud.git main --squash

# Pull updates later
git subtree pull --prefix=path/to/zotero-local-crud \
  https://github.com/DaveCBeck/zotero-local-crud.git main --squash
```

## API Endpoints

All endpoints are available at `http://localhost:23119/local-crud/...`

### Health Check

```bash
GET /local-crud/ping
```

Returns plugin status and version info.

**Response:**
```json
{
  "status": "ok",
  "plugin": "zotero-local-crud",
  "version": "1.0.0",
  "zoteroVersion": "7.0.30",
  "timestamp": "2024-12-16T12:00:00.000Z",
  "libraryID": 1
}
```

### Create Item

```bash
POST /local-crud/items
Content-Type: application/json
```

**Request Body:**
```json
{
  "itemType": "book",
  "fields": {
    "title": "Example Book",
    "date": "2024",
    "publisher": "Example Press"
  },
  "creators": [
    {
      "firstName": "John",
      "lastName": "Doe",
      "creatorType": "author"
    }
  ],
  "tags": ["tag1", "tag2"],
  "collections": ["ABCD1234"]
}
```

**Response (201):**
```json
{
  "key": "XYZ12345",
  "itemID": 123,
  "version": 0,
  "itemType": "book"
}
```

### Get/Update/Delete Item

All item operations use a single POST endpoint with an `action` field:

```bash
POST /local-crud/item
Content-Type: application/json
```

#### Get Item
```json
{
  "action": "get",
  "key": "XYZ12345"
}
```

**Response (200):**
```json
{
  "key": "XYZ12345",
  "itemID": 123,
  "itemType": "book",
  "version": 0,
  "libraryID": 1,
  "dateAdded": "2024-12-16 12:00:00",
  "dateModified": "2024-12-16 12:00:00",
  "fields": {
    "title": "Example Book",
    "date": "2024"
  },
  "creators": [...],
  "tags": [{"tag": "tag1"}],
  "collections": []
}
```

#### Update Item
```json
{
  "action": "update",
  "key": "XYZ12345",
  "fields": {
    "title": "Updated Title"
  },
  "tags": ["new-tag"]
}
```

Only include the properties you want to update. Tags, creators, and collections are replaced entirely when provided.

**Response (200):**
```json
{
  "key": "XYZ12345",
  "version": 0,
  "dateModified": "2024-12-16 13:00:00"
}
```

#### Delete Item
```json
{
  "action": "delete",
  "key": "XYZ12345"
}
```

**Response:** `204 No Content`

### Search Items

```bash
POST /local-crud/search
Content-Type: application/json
```

**Request Body:**
```json
{
  "conditions": [
    {"condition": "title", "operator": "contains", "value": "example"},
    {"condition": "tag", "operator": "is", "value": "research"}
  ],
  "limit": 100,
  "includeFullData": false
}
```

**Search Operators:**
- `is`, `isNot`
- `contains`, `doesNotContain`
- `isLessThan`, `isGreaterThan`
- `isBefore`, `isAfter`
- `beginsWith`

**Special Conditions:**
- `quicksearch-everything` - Search all fields
- `quicksearch-titleCreatorYear` - Search title, creator, and year

**Response (200):**
```json
{
  "total": 5,
  "limit": 100,
  "items": [
    {
      "key": "XYZ12345",
      "itemID": 123,
      "itemType": "book",
      "title": "Example Book",
      "dateModified": "2024-12-16 12:00:00"
    }
  ]
}
```

Set `includeFullData: true` to get complete item data in results.

## Item Types

Common item types:
- `book`, `bookSection`
- `journalArticle`, `magazineArticle`, `newspaperArticle`
- `thesis`, `report`
- `webpage`, `blogPost`
- `conferencePaper`, `presentation`
- `patent`, `statute`, `case`
- `document`, `letter`, `manuscript`

Get all valid item types by sending an invalid type - the error will list all valid types.

## Examples

### cURL Examples

```bash
# Health check
curl http://localhost:23119/local-crud/ping

# Create a book
curl -X POST http://localhost:23119/local-crud/items \
  -H "Content-Type: application/json" \
  -d '{"itemType": "book", "fields": {"title": "My Book", "date": "2024"}}'

# Get an item
curl -X POST http://localhost:23119/local-crud/item \
  -H "Content-Type: application/json" \
  -d '{"action": "get", "key": "XYZ12345"}'

# Update an item
curl -X POST http://localhost:23119/local-crud/item \
  -H "Content-Type: application/json" \
  -d '{"action": "update", "key": "XYZ12345", "fields": {"title": "New Title"}}'

# Delete an item
curl -X POST http://localhost:23119/local-crud/item \
  -H "Content-Type: application/json" \
  -d '{"action": "delete", "key": "XYZ12345"}'

# Search for items
curl -X POST http://localhost:23119/local-crud/search \
  -H "Content-Type: application/json" \
  -d '{"conditions": [{"condition": "quicksearch-everything", "value": "machine learning"}], "limit": 10}'
```

### Python Example

```python
import httpx

BASE_URL = "http://localhost:23119"

async def example():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # Create item
        response = await client.post("/local-crud/items", json={
            "itemType": "book",
            "fields": {"title": "My Book"}
        })
        key = response.json()["key"]

        # Read item
        item = await client.post("/local-crud/item", json={
            "action": "get",
            "key": key
        })
        print(item.json())

        # Update item
        await client.post("/local-crud/item", json={
            "action": "update",
            "key": key,
            "fields": {"title": "Updated Title"}
        })

        # Search
        results = await client.post("/local-crud/search", json={
            "conditions": [{"condition": "title", "value": "Updated"}]
        })
        print(results.json())

        # Delete
        await client.post("/local-crud/item", json={
            "action": "delete",
            "key": key
        })
```

## Compatibility

- **Zotero 7.x** (required)
- Port 23119 (Zotero's HTTP server)

## Security

This plugin is designed for local automation only. The Zotero HTTP server (port 23119) only listens on localhost by default and has no authentication. Do not expose this port to the network.

## License

MIT

## Contributing

Issues and pull requests welcome at the project repository.
