# =============================================================================
# watchdog/scraper.py  —  Media Acquisition / Web Scraper
# =============================================================================
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from config import RAW_VIDEO_DIR, RAW_TEXT_DIR

logger = logging.getLogger(__name__)

REQUEST_DELAY_SECONDS = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-NG,en;q=0.9",
}


@dataclass
class ScrapedArtifact:
    source_url:     str
    media_type:     str
    local_path:     Optional[Path] = None
    raw_text:       Optional[str]  = None
    publisher_name: Optional[str]  = None
    scraped_at:     datetime       = field(default_factory=lambda: datetime.now(timezone.utc))
    page_title:     Optional[str]  = None

    def is_video(self) -> bool:
        return self.media_type == "video"

    def is_text(self) -> bool:
        return self.media_type == "text"

    def __repr__(self):
        return (
            f"<ScrapedArtifact type={self.media_type} "
            f"publisher='{self.publisher_name}' "
            f"url={self.source_url[:60]}...>"
        )


def scrape_url(url: str, download_video: bool = True) -> Optional[ScrapedArtifact]:
    url = url.strip()
    logger.info(f"Scraping: {url}")

    parsed = urlparse(url)
    lower_path = parsed.path.lower()

    is_video_url = (
        lower_path.endswith((".mp4", ".avi", ".webm", ".mov")) or
        any(domain in parsed.netloc for domain in
            ["youtube.com", "youtu.be", "facebook.com", "instagram.com",
             "tiktok.com", "twitter.com", "x.com"])
    )

    if is_video_url:
        return _scrape_video(url, download=download_video)
    else:
        return _scrape_text_page(url)


def _scrape_text_page(url: str) -> Optional[ScrapedArtifact]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"HTTP request failed for {url}: {e}")
        return None

    time.sleep(REQUEST_DELAY_SECONDS)

    # BUG FIX (found via real-data inspection — see docs/PROJECT_CONTEXT.md
    # bug log): stdlib html.parser does not implement HTML5's implicit
    # <p>-closing rule. Several Nigerian news sites (confirmed: BusinessDay,
    # Punch) emit article HTML with unclosed <p> tags. html.parser then
    # nests each paragraph inside the previous one instead of treating them
    # as siblings, so find_all("p") + get_text() below recursed into every
    # nested descendant — producing a cascading duplicate ("para 1", then
    # "para 1+2", then "para 1+2+3", ...) that inflated a ~900-word
    # BusinessDay article to 11,300 words and a Punch article ~3.4x. lxml's
    # HTML parser correctly auto-closes <p> per spec, restoring flat
    # sibling paragraphs.
    soup = BeautifulSoup(response.text, "lxml")

    publisher = None
    og_site = soup.find("meta", property="og:site_name")
    if og_site:
        publisher = og_site.get("content", "").strip()
    if not publisher:
        publisher = urlparse(url).netloc.replace("www.", "").split(".")[0].title()

    title_tag = soup.find("title")
    page_title = title_tag.text.strip() if title_tag else ""

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Cross-link teasers and CTA boilerplate that show up as ordinary <p>
    # tags interleaved with real article paragraphs (e.g. "Read also: ...",
    # newsletter subscription prompts, social-channel CTAs). With the
    # html.parser nesting bug fixed above these no longer get duplicated
    # dozens of times, but they were still never part of the article body
    # and inject unrelated headline text into the middle of sentences,
    # which would corrupt burstiness/perplexity scoring.
    _BOILERPLATE_PATTERNS = re.compile(
        r"^(read also|read next|related|subscribe|join .+ (whatsapp|telegram) channel|"
        r"get newsletter updates|delivery preference)\b",
        re.IGNORECASE,
    )

    paragraphs = soup.find_all("p")
    body_text = " ".join(
        p.get_text(separator=" ").strip() for p in paragraphs
        if not _BOILERPLATE_PATTERNS.match(p.get_text(separator=" ").strip())
    )

    if len(body_text.strip()) < 50:
        body_text = soup.get_text(separator=" ")

    body_text = re.sub(r'\s+', ' ', body_text).strip()

    if not body_text:
        logger.warning(f"No text content extracted from {url}")
        return None

    logger.info(
        f"Text scraped: '{page_title[:50]}...' | "
        f"publisher='{publisher}' | {len(body_text)} chars"
    )

    return ScrapedArtifact(
        source_url=url,
        media_type="text",
        raw_text=body_text,
        publisher_name=publisher,
        page_title=page_title,
    )


def _scrape_video(url: str, download: bool = True) -> Optional[ScrapedArtifact]:
    parsed = urlparse(url)
    publisher = urlparse(url).netloc.replace("www.", "").split(".")[0].title()

    if url.lower().endswith((".mp4", ".avi", ".webm")):
        if not download:
            return ScrapedArtifact(source_url=url, media_type="video",
                                   publisher_name=publisher)
        local_path = _direct_video_download(url)
        if local_path:
            return ScrapedArtifact(source_url=url, media_type="video",
                                   local_path=local_path, publisher_name=publisher)
        return None

    if not download:
        return ScrapedArtifact(source_url=url, media_type="video",
                               publisher_name=publisher)

    local_path = _ytdlp_download(url)
    if local_path:
        logger.info(f"Video downloaded via yt-dlp: {local_path.name}")
        return ScrapedArtifact(source_url=url, media_type="video",
                               local_path=local_path, publisher_name=publisher)

    logger.warning(f"Video download failed for: {url}")
    return None


def _direct_video_download(url: str) -> Optional[Path]:
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=60)
        response.raise_for_status()
        filename = Path(urlparse(url).path).name or "video_artifact.mp4"
        save_path = RAW_VIDEO_DIR / filename
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except Exception as e:
        logger.error(f"Direct video download failed: {e}")
        return None


def _ytdlp_download(url: str) -> Optional[Path]:
    """
    Downloads a video from YouTube/social platforms using yt-dlp.
    yt-dlp is the maintained fork of youtube-dl.

    AUTHENTICATION (Facebook/Instagram require this — see config.py):
    Both platforms reject anonymous automated downloads with a
    "login required" or "no video formats found" error. If either
    YTDLP_COOKIES_FROM_BROWSER or YTDLP_COOKIES_FILE is configured, this
    function authenticates as the operator's own logged-in session before
    attempting the download. Per §1.6 scope, this only ever accesses
    content already publicly visible to that account — no private-content
    scraping, no third-party credentials.
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp not installed. Run: pip install yt-dlp")
        return None

    from config import YTDLP_COOKIES_FROM_BROWSER, YTDLP_COOKIES_FILE

    # BUG FIX (found via live run — 6 different Instagram URLs all produced
    # the identical output filename "Video by mtnng.mp4" and silently
    # overwrote each other on disk, one after another). Root cause: the
    # template used only %(title)s, and Instagram frequently returns a
    # generic, non-unique title (e.g. "Video by <account>") shared across
    # many different posts from the same account — it is NOT a reliable
    # unique identifier the way it usually is on YouTube. %(id)s is yt-dlp's
    # universal per-video unique identifier across every supported
    # extractor (YouTube, Facebook, Instagram, TikTok, etc.) — including it
    # guarantees distinct filenames regardless of how repetitive the title
    # is. This was previously silently destroying downloaded content: 6 of
    # 7 manifest rows in one run ended up pointing at the same overwritten
    # file, each with a different sample_id but identical (wrong) content.
    output_template = str(RAW_VIDEO_DIR / "%(title)s_%(id)s.%(ext)s")
    ydl_opts = {
        # BUG FIX (found via live Instagram run): a bare "best[height<=720]"
        # filter has no fallback — if a specific post/reel's format list
        # doesn't happen to include anything matching that exact filter
        # (common on Instagram, which serves a narrower/inconsistent format
        # set per-post), yt-dlp fails outright with "Requested format is
        # not available" even though OTHER usable formats do exist for that
        # same content. The "/" fallback chain tries the height-restricted
        # selector first, and only falls through to plain "best" (whatever
        # highest quality IS available) if that specific filter matches
        # nothing — this is yt-dlp's standard, documented pattern for
        # exactly this situation.
        "format": "best[height<=720]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Defense-in-depth alongside the %(id)s fix above: even if a future
        # naming collision somehow occurs, refuse to overwrite rather than
        # silently destroying a previously downloaded sample. yt-dlp skips
        # (doesn't error) when the target file already exists — the caller
        # still gets a valid Path back, just pointing at the pre-existing
        # file, so this fails safe rather than fails loud.
        "nooverwrites": True,
    }

    used_browser_auth = False
    if YTDLP_COOKIES_FROM_BROWSER:
        ydl_opts["cookiesfrombrowser"] = (YTDLP_COOKIES_FROM_BROWSER,)
        used_browser_auth = True
        logger.info(f"Authenticating via {YTDLP_COOKIES_FROM_BROWSER} browser session")
    elif YTDLP_COOKIES_FILE:
        cookies_path = Path(YTDLP_COOKIES_FILE)
        if cookies_path.exists():
            ydl_opts["cookiefile"] = str(cookies_path)
            logger.info(f"Authenticating via cookies file: {cookies_path}")
        else:
            logger.warning(
                f"YTDLP_COOKIES_FILE is set to '{cookies_path}' but that file "
                f"doesn't exist — proceeding without auth (will likely fail "
                f"on Instagram/Facebook)."
            )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return Path(filename)
    except Exception as e:
        error_str = str(e)

        # BUG FIX: "cookiesfrombrowser" auto-detection is fragile — it fails
        # outright (rather than falling back gracefully) whenever the browser
        # isn't installed at yt-dlp's expected OS-specific path. The single
        # most common cause of this in practice: running inside WSL, where
        # Chrome is installed on the WINDOWS side, so its cookie database
        # lives on the Windows filesystem, not in the Linux home directory
        # yt-dlp searches by default. This produces a confusing raw error
        # ("could not find chrome cookies database in ...") that gives no
        # hint toward the fix. Catch it specifically and point directly at
        # the more reliable alternative (a manually-exported cookies.txt),
        # which sidesteps browser-path auto-detection entirely.
        if used_browser_auth and "could not find" in error_str.lower() and "cookies database" in error_str.lower():
            logger.error(
                f"yt-dlp download failed for {url}: browser cookie auto-detection "
                f"could not locate the cookies database on disk.\n"
                f"  Raw error: {error_str}\n"
                f"  This usually means the browser you named isn't installed at "
                f"yt-dlp's expected path on THIS system — very commonly because "
                f"you're running inside WSL, where Chrome lives on the Windows "
                f"side, not in the Linux filesystem yt-dlp is searching.\n"
                f"  FIX: switch to the more reliable Option 2 instead — export a "
                f"cookies.txt file (e.g. via the 'Get cookies.txt LOCALLY' "
                f"browser extension) and set YTDLP_COOKIES_FILE=cookies.txt in "
                f".env instead of YTDLP_COOKIES_FROM_BROWSER."
            )
        else:
            logger.error(f"yt-dlp download failed for {url}: {error_str}")
        return None


def scrape_url_list(
    urls: list[str],
    download_video: bool = True
) -> list[ScrapedArtifact]:
    artifacts = []
    total = len(urls)

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{total}] Processing: {url[:80]}")
        artifact = scrape_url(url, download_video=download_video)
        if artifact:
            artifacts.append(artifact)
        else:
            logger.warning(f"[{i}/{total}] Skipped (scrape failed): {url[:80]}")
        if i < total:
            time.sleep(REQUEST_DELAY_SECONDS)

    logger.info(
        f"Scraping complete: {len(artifacts)}/{total} artifacts retrieved "
        f"({sum(1 for a in artifacts if a.is_video())} video, "
        f"{sum(1 for a in artifacts if a.is_text())} text)"
    )
    return artifacts
