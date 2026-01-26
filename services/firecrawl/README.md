# Self-hosted Firecrawl

Local web scraping service using [Firecrawl](https://github.com/firecrawl/firecrawl).

## Features

- **Scrape**: Convert web pages to markdown
- **Map**: Discover URLs on a website
- **Search**: Web search (via Google or SearXNG)

## Limitations

The self-hosted version does **not** have access to Fire-engine, which means:

- No stealth proxy (anti-bot bypass)
- No advanced IP rotation
- No robot detection bypass

When local scraping fails (e.g., due to bot detection), the code automatically falls back to:
1. Cloud Firecrawl with stealth proxy
2. Local Playwright browser

## Usage

```bash
# Start the service
./services/services.sh up

# Or start just Firecrawl
docker compose -f services/firecrawl/docker-compose.yml up -d

# Check status
curl http://localhost:3002/

# Test scrape
curl -X POST http://localhost:3002/v1/scrape \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com"}'
```

## Configuration

Set in `.env`:

```bash
# Enable local Firecrawl (default: http://localhost:3002)
FIRECRAWL_LOCAL_URL=http://localhost:3002

# Cloud API key (required for stealth fallback)
FIRECRAWL_API_KEY=fc-...

# Request timeout in seconds (default: 45)
FIRECRAWL_TIMEOUT=45

# Skip local and use cloud only (for debugging)
FIRECRAWL_SKIP_LOCAL=false
```

## SearXNG (Optional)

For fully local search without Google, uncomment the SearXNG service in `docker-compose.yml` and set:

```bash
SEARXNG_ENDPOINT=http://searxng:8080
```

## Resources

- [Firecrawl Self-Hosting Guide](https://github.com/firecrawl/firecrawl/blob/main/SELF_HOST.md)
- [Firecrawl API Docs](https://docs.firecrawl.dev/)
