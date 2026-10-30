import html
import json
from datetime import datetime, timezone

import requests

COLLECTOR_URL = "http://localhost:9000"
DELETED = {"[deleted]", "[removed]", ""}


def iso(ts) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_links(c: dict) -> list[str]:
    urls = []

    def add(u):
        if isinstance(u, str):
            u = html.unescape(u).strip()
            if u.startswith(("http://", "https://")) and u not in urls:
                urls.append(u)

    add(c.get("url_overridden_by_dest"))
    add(c.get("url"))
    add(c.get("thumbnail"))
    for im in (c.get("preview") or {}).get("images", []):
        add((im.get("source") or {}).get("url"))
        for v in (im.get("variants") or {}).values():
            add((v.get("source") or {}).get("url"))
    for m in (c.get("media_metadata") or {}).values():
        if isinstance(m, dict):
            s = m.get("s") or {}
            add(s.get("u") or s.get("gif") or s.get("mp4"))
    for key in ("secure_media", "media"):
        mm = c.get(key) or {}
        add((mm.get("reddit_video") or {}).get("fallback_url"))
        add((mm.get("oembed") or {}).get("url"))
    return urls


def build_item(c: dict) -> dict | None:
    is_post = c.get("name", "").startswith("t3_") or "title" in c and "body" not in c
    author = c.get("author") or ""
    title = c.get("title") or ""
    text = c.get("selftext") if is_post else c.get("body")
    text = (text or "").strip()
    content = f"{title}\n{text}".strip() if is_post else text
    if text in DELETED and not title or not content:
        return None

    ext_id = c.get("name") or (("t3_" if is_post else "t1_") + c["id"])
    permalink = c.get("permalink") or ""
    url = permalink if permalink.startswith("http") else "https://www.reddit.com" + permalink

    links = [u for u in extract_links(c) if u != url and u not in content and html.unescape(content).find(u) < 0]
    if links:
        content += "\n" + "\n".join("--" + u for u in links)

    summary = {
        "score": str(c.get("score", c.get("ups", ""))),
        "upvote_ratio": str(c.get("upvote_ratio", "")),
        "num_comments": str(c.get("num_comments", "")),
        "communityName": "r/" + str(c.get("subreddit", "")),
        "data_type": "post" if is_post else "comment",
    }

    item = {
        "content": content,
        "created_at": iso(c["created_utc"]),
        "domain": "reddit.com",
        "url": url,
        "external_id": ext_id,
        "summary": json.dumps(summary, ensure_ascii=False),
    }
    if is_post:
        item["title"] = title
    elif c.get("parent_id"):
        item["external_parent_id"] = c["parent_id"]
    if author not in DELETED:
        item["author"] = author
        item["username"] = author
        item["userprofile_url"] = f"https://www.reddit.com/user/{author}"
    pic = c.get("url_overridden_by_dest") or c.get("thumbnail") or ""
    if is_post and pic.startswith(("http://", "https://")):
        item["picture"] = pic
    return item


def post_item(item: dict) -> dict:
    r = requests.post(f"{COLLECTOR_URL}/store_item", json=item, timeout=10)
    r.raise_for_status()
    return r.json()


def post_items(items: list[dict]) -> dict:
    r = requests.post(f"{COLLECTOR_URL}/store_items", json=items, timeout=30)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    with open("comments.json", encoding="utf-8") as f:
        raw = json.load(f)
    items = [i for i in (build_item(c) for c in raw) if i]
    print(post_items(items))
