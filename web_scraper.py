"""Asynchronous web and RSS scraper for news ingestion."""
import asyncio
import logging
from typing import List, Optional
import xml.etree.ElementTree as ET

import aiohttp
from bs4 import BeautifulSoup

log = logging.getLogger("scraper")

async def fetch_url(session: aiohttp.ClientSession, url: str, use_playwright: bool = False) -> Optional[str]:
    """Fetch text content from a URL."""
    if use_playwright:
        return await _fetch_playwright(url)
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with session.get(url, headers=headers, timeout=15) as resp:
            resp.raise_for_status()
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove script/style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            # Extract main text
            text = soup.get_text(separator=" ", strip=True)
            return text if len(text) > 100 else None
    except Exception as e:
        log.warning("Failed to fetch %s: %s", url, e)
        return None

async def _fetch_playwright(url: str) -> Optional[str]:
    """Fallback for JS-heavy sites requiring rendering."""
    from playwright.async_api import async_playwright
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            text = await page.evaluate("document.body.innerText")
            await browser.close()
            return text if len(text) > 100 else None
    except Exception as e:
        log.warning("Playwright failed on %s: %s", url, e)
        return None

async def parse_rss_feed(session: aiohttp.ClientSession, rss_url: str) -> List[dict]:
    """Parse an RSS feed and return a list of {title, link, description}."""
    articles = []
    try:
        async with session.get(rss_url, timeout=15) as resp:
            resp.raise_for_status()
            xml_data = await resp.text()
        
        root = ET.fromstring(xml_data)
        for item in root.findall(".//item"):
            title = item.find("title")
            link = item.find("link")
            desc = item.find("description")
            
            if title is not None and link is not None:
                articles.append({
                    "title": title.text or "",
                    "url": link.text or "",
                    "description": desc.text or "" if desc is not None else ""
                })
        log.info("Found %d articles in RSS feed: %s", len(articles), rss_url)
    except Exception as e:
        log.error("Failed to parse RSS feed %s: %s", rss_url, e)
    
    return articles