# Substack Publish Utility

Publish illustrated markdown to Substack with native footnotes (hover-to-preview).

## Features

- **Native footnotes**: Converts `[@KEY]` citations to Substack's hover-preview footnotes
- **Image uploads**: Automatically uploads local images to Substack's S3
- **Multiple publications**: Supports publishing to any of your Substack publications
- **Paywall support**: Use `---paywall---` to mark where free preview ends
- **Audience control**: Restrict posts to paid subscribers, founding members, etc.
- **Markdown cleanup**: Strips YAML frontmatter, duplicate titles, fixes links and horizontal rules

## Quick Start

```bash
python scripts/substack_publish.py article.md \
  --title "My Article" \
  --publication mysubstack.substack.com \
  --draft-only
```

## Setup

### 1. Get Substack cookies

Export cookies while logged into Substack:

1. Install [Cookie-Editor](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm) browser extension
2. Go to substack.com (logged in)
3. Click extension → Export → Export as JSON
4. Save to `.substack-cookies.json` in project root

Or manually create the file:
```json
{
  "substack.lli": "your_lli_value",
  "substack.sid": "your_sid_value"
}
```

### 2. Install dependency

```bash
pip install python-substack
```

## CLI Usage

```bash
# Create draft (recommended for testing)
python scripts/substack_publish.py article.md \
  --title "Article Title" \
  --subtitle "Optional subtitle" \
  --publication davecbeck.substack.com \
  --draft-only

# Publish immediately
python scripts/substack_publish.py article.md \
  --title "Article Title" \
  --publication davecbeck.substack.com

# Specify custom cookies path
python scripts/substack_publish.py article.md \
  --title "Article Title" \
  --publication davecbeck.substack.com \
  --cookies ~/.my-substack-cookies.json
```

### Options

| Flag | Description |
|------|-------------|
| `--title` | Post title (required) |
| `--subtitle` | Post subtitle |
| `--publication` | Publication subdomain (e.g., `mysubstack.substack.com`) |
| `--draft-only` | Create draft without publishing |
| `--cookies` | Path to cookies JSON file |
| `--audience` | `everyone`, `only_paid`, `founding`, `only_free` |
| `-v, --verbose` | Enable debug logging |

## Programmatic Usage

```python
from utils.substack_publish import SubstackPublisher, SubstackConfig

config = SubstackConfig(
    cookies_path=".substack-cookies.json",
    publication_url="davecbeck.substack.com",
    audience="everyone",
)

publisher = SubstackPublisher(config)

# Create draft
result = publisher.create_draft(
    markdown=article_content,
    title="My Article",
    subtitle="Optional subtitle",
)
print(f"Draft: {result['draft_url']}")

# Or publish immediately
result = publisher.publish_post(
    markdown=article_content,
    title="My Article",
)
print(f"Published: {result['publish_url']}")
```

## Expected Markdown Format

The utility expects markdown from the `evening_reads` + `illustrate` workflows:

```markdown
---
title: Article Title
status: success
---

# Article Title

![Header image](/path/to/local/image.jpg)

*Photo by Someone via [Pexels](https://pexels.com/...)*

Body text with citations [@CITATION_KEY].

---paywall---

## Section Header (paid content below here)

More content [@ANOTHER_KEY; @THIRD_KEY].

---

## References

[@CITATION_KEY] Author, A. (2024). Title. *Journal*.

[@ANOTHER_KEY] Author, B. (2024). Another paper. *Journal*.
```

### What gets transformed

| Input | Output |
|-------|--------|
| YAML frontmatter | Stripped |
| `# Title` at top | Stripped (title in post metadata) |
| `[@KEY]` citations | Footnote superscripts (hover to preview) |
| `## References` section | Stripped (content moved to footnotes) |
| `---` horizontal rules | Substack dividers |
| `---paywall---` | Paywall boundary (free preview ends here) |
| `[text](url)` links | Clickable links |
| Local image paths | Uploaded to S3, URLs replaced |

## Paywalls and Audience Control

### Paywall cutoff

Add `---paywall---` on its own line where the free preview should end:

```markdown
This content is visible to everyone.

---paywall---

This content is only visible to paid subscribers.
```

Free subscribers see everything above the marker; paid content appears below.

### Audience options

Control who can see the entire post:

| Value | Who can read |
|-------|--------------|
| `everyone` | All subscribers (default) |
| `only_paid` | Paid subscribers only |
| `founding` | Founding members only |
| `only_free` | Free subscribers only |

```bash
# Paid-only post (no free preview)
python scripts/substack_publish.py article.md \
  --title "Premium Content" \
  --audience only_paid

# Free preview with paywall cutoff
python scripts/substack_publish.py article.md \
  --title "Article with Preview" \
  --audience everyone
# (use ---paywall--- in the markdown to mark the cutoff)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SUBSTACK_COOKIES_PATH` | Default cookies file path |
| `SUBSTACK_PUBLICATION_URL` | Default publication URL |
