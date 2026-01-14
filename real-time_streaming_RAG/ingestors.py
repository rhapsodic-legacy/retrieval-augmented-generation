"""
Stream Ingestors

Ingests data from various streaming sources:
- RSS/Atom feeds (news)
- Webhooks
- Log files
- Simulated data (for demo)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, AsyncGenerator
from datetime import datetime, timezone
import asyncio
import threading
import hashlib
import random
import time

from .content_types import (
    StreamItem,
    NewsItem,
    LogEntry,
    SocialPost,
    ContentType,
    ContentPriority,
    create_stream_item,
)


class BaseIngestor(ABC):
    """Abstract base class for stream ingestors."""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.callback: Optional[Callable[[StreamItem], None]] = None
        self._running = False
    
    def set_callback(self, callback: Callable[[StreamItem], None]):
        """Set callback for new items."""
        self.callback = callback
    
    @abstractmethod
    def start(self):
        """Start ingesting."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop ingesting."""
        pass
    
    def _emit(self, item: StreamItem):
        """Emit an item via callback."""
        if self.callback:
            self.callback(item)


class RSSIngestor(BaseIngestor):
    """
    Ingest from RSS/Atom feeds.
    
    Polls feeds at regular intervals and emits new items.
    """
    
    def __init__(
        self,
        feed_urls: list[str],
        source_name: str = "rss",
        poll_interval: int = 300,  # 5 minutes
    ):
        super().__init__(source_name)
        self.feed_urls = feed_urls
        self.poll_interval = poll_interval
        self.seen_ids: set[str] = set()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self):
        """Start polling feeds."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop polling."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
    
    def _poll_loop(self):
        """Polling loop."""
        while not self._stop_event.is_set():
            for url in self.feed_urls:
                try:
                    self._poll_feed(url)
                except Exception as e:
                    print(f"Error polling {url}: {e}")
            
            self._stop_event.wait(self.poll_interval)
    
    def _poll_feed(self, url: str):
        """Poll a single feed."""
        try:
            import feedparser
        except ImportError:
            print("feedparser not installed. Install with: pip install feedparser")
            return
        
        feed = feedparser.parse(url)
        
        for entry in feed.entries:
            # Generate unique ID
            entry_id = entry.get('id') or entry.get('link') or hashlib.md5(
                entry.get('title', '').encode()
            ).hexdigest()
            
            if entry_id in self.seen_ids:
                continue
            
            self.seen_ids.add(entry_id)
            
            # Parse timestamp
            timestamp = datetime.now(timezone.utc)
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                timestamp = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                timestamp = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            
            # Extract content
            content = entry.get('summary') or entry.get('description') or entry.get('title', '')
            
            # Strip HTML
            import re
            content = re.sub(r'<[^>]+>', '', content)
            
            item = NewsItem(
                id=entry_id[:16],
                content=content[:2000],
                content_type=ContentType.NEWS,
                timestamp=timestamp,
                source=self.source_name,
                source_url=entry.get('link'),
                title=entry.get('title'),
                author=entry.get('author'),
                category=feed.feed.get('title'),
            )
            
            self._emit(item)


class WebhookIngestor(BaseIngestor):
    """
    Receive data via webhooks.
    
    Exposes methods to be called by the API server.
    """
    
    def __init__(self, source_name: str = "webhook"):
        super().__init__(source_name)
    
    def start(self):
        self._running = True
    
    def stop(self):
        self._running = False
    
    def ingest(self, data: dict) -> Optional[StreamItem]:
        """
        Ingest a webhook payload.
        
        Expected format:
        {
            "content": "text content",
            "type": "news|log|social|generic",
            "title": "optional title",
            "source": "optional source override",
            "timestamp": "optional ISO timestamp",
            "metadata": {}
        }
        """
        if not self._running:
            return None
        
        content = data.get("content")
        if not content:
            return None
        
        # Parse timestamp
        timestamp = datetime.now(timezone.utc)
        if data.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except ValueError:
                pass
        
        item = create_stream_item(
            content=content,
            content_type=data.get("type", "generic"),
            source=data.get("source", self.source_name),
            timestamp=timestamp,
            title=data.get("title"),
            metadata=data.get("metadata", {}),
        )
        
        self._emit(item)
        return item


class LogFileIngestor(BaseIngestor):
    """
    Tail log files for new entries.
    """
    
    def __init__(
        self,
        file_path: str,
        source_name: str = "logs",
        poll_interval: float = 1.0,
    ):
        super().__init__(source_name)
        self.file_path = file_path
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._position = 0
    
    def start(self):
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._tail_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
    
    def _tail_loop(self):
        """Tail the log file."""
        import os
        
        # Start at end of file
        try:
            with open(self.file_path, 'r') as f:
                f.seek(0, 2)  # End of file
                self._position = f.tell()
        except FileNotFoundError:
            self._position = 0
        
        while not self._stop_event.is_set():
            try:
                with open(self.file_path, 'r') as f:
                    f.seek(self._position)
                    
                    for line in f:
                        line = line.strip()
                        if line:
                            self._process_log_line(line)
                    
                    self._position = f.tell()
                    
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error reading log file: {e}")
            
            self._stop_event.wait(self.poll_interval)
    
    def _process_log_line(self, line: str):
        """Process a log line."""
        # Try to parse common log formats
        level = "INFO"
        timestamp = datetime.now(timezone.utc)
        
        # Check for log level
        for lvl in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            if lvl in line.upper():
                level = lvl
                break
        
        item = LogEntry(
            id=hashlib.md5(line.encode()).hexdigest()[:16],
            content=line,
            content_type=ContentType.LOG,
            timestamp=timestamp,
            source=self.source_name,
            level=level,
        )
        
        self._emit(item)


class SimulatedNewsIngestor(BaseIngestor):
    """
    Generates simulated news for demo purposes.
    """
    
    HEADLINES = [
        ("Tech Giants Report Record Quarterly Earnings", "business"),
        ("New AI Model Breaks Performance Records", "technology"),
        ("Climate Summit Reaches Historic Agreement", "world"),
        ("Stock Markets Rally on Economic Data", "business"),
        ("Scientists Discover New Species in Amazon", "science"),
        ("Central Bank Announces Interest Rate Decision", "business"),
        ("Major Security Vulnerability Found in Popular Software", "technology"),
        ("Elections Results: Early Returns Coming In", "politics"),
        ("Sports Team Wins Championship in Overtime", "sports"),
        ("New Study Reveals Health Benefits of Exercise", "health"),
        ("Space Agency Announces Mars Mission Update", "science"),
        ("Entertainment Awards Show Winners Announced", "entertainment"),
        ("Energy Prices Surge Amid Supply Concerns", "business"),
        ("Breakthrough in Renewable Energy Technology", "technology"),
        ("International Trade Deal Negotiations Progress", "world"),
    ]
    
    def __init__(
        self,
        source_name: str = "news-sim",
        items_per_minute: float = 5,
    ):
        super().__init__(source_name)
        self.items_per_minute = items_per_minute
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self):
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
    
    def _generate_loop(self):
        """Generate news items."""
        interval = 60 / self.items_per_minute
        
        while not self._stop_event.is_set():
            self._generate_item()
            self._stop_event.wait(interval + random.uniform(-0.5, 0.5) * interval)
    
    def _generate_item(self):
        """Generate a single news item."""
        headline, category = random.choice(self.HEADLINES)
        
        # Add some variation
        variations = [
            f"Breaking: {headline}",
            f"Update: {headline}",
            f"Analysis: {headline}",
            headline,
        ]
        title = random.choice(variations)
        
        # Generate content
        content = f"{title}. " + self._generate_content(category)
        
        item = NewsItem(
            id=hashlib.md5(f"{title}{time.time()}".encode()).hexdigest()[:16],
            content=content,
            content_type=ContentType.NEWS,
            timestamp=datetime.now(timezone.utc),
            source=self.source_name,
            title=title,
            category=category,
            priority=random.choice(list(ContentPriority)),
        )
        
        self._emit(item)
    
    def _generate_content(self, category: str) -> str:
        """Generate fake article content."""
        templates = {
            "business": "Market analysts are closely watching developments as investors assess the implications for the broader economy. Industry experts suggest this could signal a shift in market dynamics.",
            "technology": "The development represents a significant advancement in the field, with potential applications across multiple industries. Researchers are optimistic about future possibilities.",
            "world": "International observers are monitoring the situation closely as diplomatic efforts continue. The outcome could have far-reaching consequences for global relations.",
            "science": "This discovery adds to our understanding of the natural world and opens new avenues for research. Scientists plan to conduct further studies.",
            "politics": "Political analysts are weighing in on the implications for upcoming elections. Voter sentiment appears to be shifting according to recent polls.",
            "sports": "Fans are celebrating the victory after an intense competition. Team officials expressed satisfaction with the performance.",
            "health": "Healthcare professionals are reviewing the findings, which could influence treatment recommendations. More research is expected.",
            "entertainment": "Industry insiders predict this will have a significant impact on upcoming releases. Fan reactions have been overwhelmingly positive.",
        }
        return templates.get(category, "Details are still emerging as the story develops.")


class SimulatedSocialIngestor(BaseIngestor):
    """
    Generates simulated social media posts for demo.
    """
    
    TOPICS = [
        "AI", "climate", "stocks", "tech", "health",
        "sports", "politics", "crypto", "startups", "gaming"
    ]
    
    SENTIMENTS = [
        "This is incredible news about {topic}! 🚀",
        "Can't believe what's happening with {topic} today",
        "My thoughts on {topic}: it's more complex than people think",
        "Breaking: major developments in {topic} space",
        "Hot take: {topic} is overrated/underrated",
        "Just read an interesting article about {topic}",
        "{topic} is trending and here's why it matters",
        "Thread 🧵 about {topic} and what it means for us",
    ]
    
    def __init__(
        self,
        source_name: str = "social-sim",
        items_per_minute: float = 10,
    ):
        super().__init__(source_name)
        self.items_per_minute = items_per_minute
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self):
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
    
    def _generate_loop(self):
        interval = 60 / self.items_per_minute
        
        while not self._stop_event.is_set():
            self._generate_item()
            self._stop_event.wait(interval + random.uniform(-0.3, 0.3) * interval)
    
    def _generate_item(self):
        topic = random.choice(self.TOPICS)
        template = random.choice(self.SENTIMENTS)
        content = template.format(topic=topic)
        
        # Add hashtags
        content += f" #{topic} #trending"
        
        item = SocialPost(
            id=hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:16],
            content=content,
            content_type=ContentType.SOCIAL,
            timestamp=datetime.now(timezone.utc),
            source=self.source_name,
            author=f"@user_{random.randint(1000, 9999)}",
            platform="twitter",
            likes=random.randint(0, 1000),
            shares=random.randint(0, 200),
            replies=random.randint(0, 50),
        )
        
        self._emit(item)
