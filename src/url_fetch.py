"""
Fetch and extract readable article text from a URL — stdlib only.

Deliberately avoids heavy scraping libraries (newspaper3k, trafilatura) so it
installs and runs anywhere. Uses a small HTMLParser that keeps paragraph-level
text and drops script/style/nav boilerplate. Good enough to feed the classifier
and summarizer; not a full readability engine.
"""
import os
import re
import ssl
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

USER_AGENT = "Mozilla/5.0 (compatible; FakeNewsDetector/1.0; +student-project)"


def _ssl_context():
    """
    Build an SSL context with a working CA bundle.

    macOS Python.framework installs often lack system CA certs (SSL
    CERTIFICATE_VERIFY_FAILED). Prefer certifi's bundle when available. Set
    FAKE_NEWS_INSECURE_SSL=1 only as a last resort — it disables verification.
    """
    if os.environ.get("FAKE_NEWS_INSECURE_SSL") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()
_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form"}
_BLOCK_TAGS = {"p", "br", "div", "li", "h1", "h2", "h3", "article", "section"}


class _ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._chunks = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        else:
            text = data.strip()
            if text:
                self._chunks.append(text + " ")

    def text(self):
        raw = "".join(self._chunks)
        # Keep paragraph-ish lines; drop tiny fragments (menus, buttons).
        lines = [ln.strip() for ln in raw.split("\n")]
        kept = [ln for ln in lines if len(ln.split()) >= 6]
        return re.sub(r"[ \t]+", " ", "\n".join(kept)).strip()


class FetchError(Exception):
    """Raised when a URL cannot be fetched or yields no usable text."""


def fetch_article(url, timeout=15, max_bytes=2_000_000):
    """
    Return (title, text) for the article at `url`.
    Raises FetchError on network problems or if no article text is found.
    """
    if "://" not in url:
        url = "https://" + url
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "html" not in ctype and "text" not in ctype:
                raise FetchError(f"URL is not an HTML page (Content-Type: {ctype}).")
            raw = resp.read(max_bytes)
    except HTTPError as e:
        raise FetchError(f"Server returned HTTP {e.code} for {url}.") from e
    except ssl.SSLError as e:
        raise FetchError(
            f"SSL verification failed for {url} ({e}). Install certificates "
            "(pip install certifi) or set FAKE_NEWS_INSECURE_SSL=1 to bypass."
        ) from e
    except (URLError, TimeoutError, ValueError) as e:
        raise FetchError(f"Could not reach {url}: {e}") from e

    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", ctype)
    if m:
        charset = m.group(1)
    html = raw.decode(charset, errors="replace")

    parser = _ArticleParser()
    parser.feed(html)
    text = parser.text()
    title = re.sub(r"\s+", " ", parser.title).strip()

    if len(text.split()) < 25:
        raise FetchError(
            "Fetched the page but couldn't extract enough article text. "
            "Try pasting the article body directly."
        )
    return title, text


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python src/url_fetch.py <url>")
        raise SystemExit(1)
    t, body = fetch_article(sys.argv[1])
    print("TITLE:", t)
    print("WORDS:", len(body.split()))
    print(body[:500])
