#!/usr/bin/env python3
"""
Self-Correcting Research Assistant

An intelligent agent that searches the web, cross-references sources,
detects contradictions, and automatically re-verifies disputed facts.

Requirements:
    pip install anthropic

Usage:
    python self_correcting_research_assistant.py "Your research query here"
    
    Or import and use programmatically:
        from self_correcting_research_assistant import ResearchAssistant
        assistant = ResearchAssistant()
        results = assistant.research("Your query")
"""

import anthropic
import json
import re
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# ============================================================================
# PROMPTS
# ============================================================================

INITIAL_RESEARCH_PROMPT = """You are a meticulous research assistant that searches the web to find accurate information. Your task is to:

1. Search for information on the given topic
2. Analyze all sources for factual claims
3. Identify any contradictions or inconsistencies between sources
4. Flag claims that need verification

IMPORTANT: Structure your response as JSON with this exact format:
{
  "findings": [
    {
      "claim": "The factual claim",
      "sources": ["source names/urls"],
      "confidence": "high|medium|low",
      "notes": "Any relevant context"
    }
  ],
  "contradictions": [
    {
      "topic": "What the contradiction is about",
      "claim_a": "First version of the claim",
      "source_a": "Source for claim A",
      "claim_b": "Contradicting version",
      "source_b": "Source for claim B"
    }
  ],
  "summary": "A brief summary of the research findings",
  "verification_needed": ["List of specific queries to run to resolve contradictions"]
}

Be thorough and skeptical. Flag anything that seems inconsistent across sources."""

VERIFICATION_PROMPT = """You are verifying potentially contradictory information. Previous research found conflicting claims that need resolution.

Search for authoritative sources to determine which claim is more accurate. Look for:
- Official sources (government, academic, company websites)
- Recent news from reputable outlets
- Multiple corroborating sources

Structure your response as JSON:
{
  "resolved_claims": [
    {
      "original_contradiction": "What was contradictory",
      "verified_fact": "The accurate information after verification",
      "explanation": "How you determined this",
      "confidence": "high|medium|low",
      "sources": ["authoritative sources used"]
    }
  ],
  "still_uncertain": [
    {
      "topic": "Topic still unclear",
      "reason": "Why it couldn't be resolved",
      "recommendation": "What additional research might help"
    }
  ],
  "additional_findings": "Any other relevant information discovered"
}"""


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Finding:
    """A single factual claim from research."""
    claim: str
    sources: list[str]
    confidence: str  # high, medium, low
    notes: str = ""


@dataclass
class Contradiction:
    """A contradiction found between sources."""
    topic: str
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str


@dataclass
class ResolvedClaim:
    """A contradiction that has been resolved through verification."""
    original_contradiction: str
    verified_fact: str
    explanation: str
    confidence: str
    sources: list[str]


@dataclass
class UncertainItem:
    """An item that couldn't be resolved."""
    topic: str
    reason: str
    recommendation: str


@dataclass
class ResearchResult:
    """Complete research results."""
    query: str
    findings: list[Finding] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    resolved_claims: list[ResolvedClaim] = field(default_factory=list)
    still_uncertain: list[UncertainItem] = field(default_factory=list)
    summary: str = ""
    iterations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# RESEARCH ASSISTANT
# ============================================================================

class ResearchAssistant:
    """
    Self-correcting research assistant that uses Claude with web search
    to find, verify, and cross-check information.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        max_verification_rounds: int = 3,
        verbose: bool = True
    ):
        """
        Initialize the research assistant.
        
        Args:
            model: Claude model to use
            max_tokens: Maximum tokens for responses
            max_verification_rounds: Maximum verification iterations
            verbose: Whether to print progress logs
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.max_verification_rounds = max_verification_rounds
        self.verbose = verbose
    
    def _log(self, message: str, level: str = "info"):
        """Print a log message if verbose mode is enabled."""
        if not self.verbose:
            return
        
        icons = {
            "info": "ℹ️ ",
            "search": "🔍",
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌",
            "start": "🚀",
            "complete": "🎉"
        }
        icon = icons.get(level, "  ")
        print(f"{icon} {message}")
    
    def _call_api(self, user_prompt: str, system_prompt: str) -> str:
        """
        Make an API call to Claude with web search enabled.
        
        Args:
            user_prompt: The user's message
            system_prompt: System instructions
            
        Returns:
            The text content of Claude's response
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        # Extract text from response blocks
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        
        return "\n".join(text_parts)
    
    def _parse_json(self, text: str) -> Optional[dict]:
        """
        Extract and parse JSON from a text response.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                self._log(f"JSON parse error: {e}", "error")
                return None
        return None
    
    def _run_initial_research(self, query: str) -> Optional[dict]:
        """
        Run the initial research phase.
        
        Args:
            query: The research query
            
        Returns:
            Parsed research results or None
        """
        self._log("Conducting initial research...", "search")
        
        prompt = f"""Research this topic thoroughly: {query}

Search multiple sources and identify any contradictions or inconsistencies."""
        
        response = self._call_api(prompt, INITIAL_RESEARCH_PROMPT)
        return self._parse_json(response)
    
    def _run_verification(self, query: str, contradictions: list[dict]) -> Optional[dict]:
        """
        Run verification to resolve contradictions.
        
        Args:
            query: The original research query
            contradictions: List of contradictions to verify
            
        Returns:
            Parsed verification results or None
        """
        self._log("Running verification to resolve contradictions...", "search")
        
        contradiction_summary = "\n".join(
            f"- {c['topic']}: \"{c['claim_a']}\" vs \"{c['claim_b']}\""
            for c in contradictions
        )
        
        prompt = f"""Verify these contradictory claims found in research about "{query}":

{contradiction_summary}

Search for authoritative sources to determine which claims are accurate."""
        
        response = self._call_api(prompt, VERIFICATION_PROMPT)
        return self._parse_json(response)
    
    def research(self, query: str) -> ResearchResult:
        """
        Conduct self-correcting research on a topic.
        
        This method:
        1. Searches for information from multiple sources
        2. Identifies contradictions between sources
        3. Re-runs verification searches to resolve contradictions
        4. Returns consolidated, verified results
        
        Args:
            query: The research topic or question
            
        Returns:
            ResearchResult containing findings, resolved contradictions, etc.
        """
        result = ResearchResult(query=query)
        
        self._log(f"Starting research on: \"{query}\"", "start")
        
        # Phase 1: Initial Research
        initial_data = self._run_initial_research(query)
        
        if not initial_data:
            self._log("Failed to parse initial research results", "error")
            return result
        
        result.iterations = 1
        result.summary = initial_data.get("summary", "")
        
        # Parse findings
        for f in initial_data.get("findings", []):
            result.findings.append(Finding(
                claim=f.get("claim", ""),
                sources=f.get("sources", []),
                confidence=f.get("confidence", "medium"),
                notes=f.get("notes", "")
            ))
        
        self._log(f"Found {len(result.findings)} claims from initial research", "success")
        
        # Parse contradictions
        contradictions = initial_data.get("contradictions", [])
        for c in contradictions:
            result.contradictions.append(Contradiction(
                topic=c.get("topic", ""),
                claim_a=c.get("claim_a", ""),
                source_a=c.get("source_a", ""),
                claim_b=c.get("claim_b", ""),
                source_b=c.get("source_b", "")
            ))
        
        # Phase 2: Verification (if contradictions found)
        if contradictions:
            self._log(
                f"Detected {len(contradictions)} contradiction(s) - initiating verification",
                "warning"
            )
            
            verification_round = 0
            remaining_contradictions = contradictions
            
            while remaining_contradictions and verification_round < self.max_verification_rounds:
                verification_round += 1
                result.iterations += 1
                
                self._log(f"Verification round {verification_round}...", "search")
                
                verification_data = self._run_verification(query, remaining_contradictions)
                
                if not verification_data:
                    self._log("Failed to parse verification results", "error")
                    break
                
                # Parse resolved claims
                for r in verification_data.get("resolved_claims", []):
                    result.resolved_claims.append(ResolvedClaim(
                        original_contradiction=r.get("original_contradiction", ""),
                        verified_fact=r.get("verified_fact", ""),
                        explanation=r.get("explanation", ""),
                        confidence=r.get("confidence", "medium"),
                        sources=r.get("sources", [])
                    ))
                
                # Parse still uncertain items
                still_uncertain = verification_data.get("still_uncertain", [])
                for u in still_uncertain:
                    result.still_uncertain.append(UncertainItem(
                        topic=u.get("topic", ""),
                        reason=u.get("reason", ""),
                        recommendation=u.get("recommendation", "")
                    ))
                
                # Check if we need another round
                if not still_uncertain:
                    break
                
                # Convert uncertain items back to contradictions format for next round
                remaining_contradictions = [
                    {"topic": u.get("topic", ""), "claim_a": "Unknown", "claim_b": "Unknown"}
                    for u in still_uncertain
                ]
            
            resolved_count = len(result.resolved_claims)
            uncertain_count = len(result.still_uncertain)
            
            self._log(f"Verification complete - resolved {resolved_count} contradiction(s)", "success")
            
            if uncertain_count > 0:
                self._log(f"{uncertain_count} item(s) still need further research", "warning")
        else:
            self._log("No contradictions detected", "success")
        
        self._log("Research process finished", "complete")
        
        return result
    
    def research_to_dict(self, query: str) -> dict:
        """
        Conduct research and return results as a dictionary.
        
        Args:
            query: The research topic or question
            
        Returns:
            Dictionary containing all research results
        """
        result = self.research(query)
        
        return {
            "query": result.query,
            "timestamp": result.timestamp,
            "iterations": result.iterations,
            "summary": result.summary,
            "findings": [
                {
                    "claim": f.claim,
                    "sources": f.sources,
                    "confidence": f.confidence,
                    "notes": f.notes
                }
                for f in result.findings
            ],
            "contradictions_found": [
                {
                    "topic": c.topic,
                    "claim_a": c.claim_a,
                    "source_a": c.source_a,
                    "claim_b": c.claim_b,
                    "source_b": c.source_b
                }
                for c in result.contradictions
            ],
            "resolved_claims": [
                {
                    "original_contradiction": r.original_contradiction,
                    "verified_fact": r.verified_fact,
                    "explanation": r.explanation,
                    "confidence": r.confidence,
                    "sources": r.sources
                }
                for r in result.resolved_claims
            ],
            "still_uncertain": [
                {
                    "topic": u.topic,
                    "reason": u.reason,
                    "recommendation": u.recommendation
                }
                for u in result.still_uncertain
            ]
        }


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_results(result: ResearchResult):
    """Print research results in a formatted way."""
    
    # Header
    print("\n" + "=" * 70)
    print("🔬 SELF-CORRECTING RESEARCH ASSISTANT - RESULTS")
    print("=" * 70)
    
    print(f"\n📋 Query: {result.query}")
    print(f"🔄 Search Iterations: {result.iterations}")
    print(f"📊 Claims Found: {len(result.findings)}")
    print(f"⚠️  Contradictions Detected: {len(result.contradictions)}")
    print(f"✅ Contradictions Resolved: {len(result.resolved_claims)}")
    
    # Summary
    if result.summary:
        print("\n" + "-" * 70)
        print("📝 SUMMARY")
        print("-" * 70)
        print(result.summary)
    
    # Findings
    if result.findings:
        print("\n" + "-" * 70)
        print("📌 FINDINGS")
        print("-" * 70)
        
        for i, finding in enumerate(result.findings, 1):
            confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                finding.confidence, "⚪"
            )
            print(f"\n{i}. {finding.claim}")
            print(f"   {confidence_emoji} Confidence: {finding.confidence}")
            if finding.sources:
                print(f"   📎 Sources: {', '.join(finding.sources)}")
            if finding.notes:
                print(f"   💡 Notes: {finding.notes}")
    
    # Contradictions
    if result.contradictions:
        print("\n" + "-" * 70)
        print("⚠️  CONTRADICTIONS DETECTED")
        print("-" * 70)
        
        for i, contradiction in enumerate(result.contradictions, 1):
            print(f"\n{i}. {contradiction.topic}")
            print(f"   Claim A: \"{contradiction.claim_a}\"")
            print(f"   Source A: {contradiction.source_a}")
            print(f"   Claim B: \"{contradiction.claim_b}\"")
            print(f"   Source B: {contradiction.source_b}")
    
    # Resolved Claims
    if result.resolved_claims:
        print("\n" + "-" * 70)
        print("✅ VERIFIED RESOLUTIONS")
        print("-" * 70)
        
        for i, resolved in enumerate(result.resolved_claims, 1):
            confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                resolved.confidence, "⚪"
            )
            print(f"\n{i}. Original Issue: {resolved.original_contradiction}")
            print(f"   ✓ Verified Fact: {resolved.verified_fact}")
            print(f"   {confidence_emoji} Confidence: {resolved.confidence}")
            print(f"   📖 Explanation: {resolved.explanation}")
            if resolved.sources:
                print(f"   📎 Sources: {', '.join(resolved.sources)}")
    
    # Still Uncertain
    if result.still_uncertain:
        print("\n" + "-" * 70)
        print("❓ STILL UNCERTAIN")
        print("-" * 70)
        
        for i, uncertain in enumerate(result.still_uncertain, 1):
            print(f"\n{i}. {uncertain.topic}")
            print(f"   Reason: {uncertain.reason}")
            print(f"   💡 Recommendation: {uncertain.recommendation}")
    
    print("\n" + "=" * 70)
    print("Research complete!")
    print("=" * 70 + "\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Self-Correcting Research Assistant - "
                    "Search, verify, and cross-check information using AI"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="The research topic or question"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum verification rounds (default: 3)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs"
    )
    
    args = parser.parse_args()
    
    # Interactive mode if no query provided
    if not args.query:
        print("\n🔬 Self-Correcting Research Assistant")
        print("-" * 40)
        args.query = input("Enter your research query: ").strip()
        
        if not args.query:
            print("No query provided. Exiting.")
            return
    
    # Create assistant and run research
    assistant = ResearchAssistant(
        model=args.model,
        max_verification_rounds=args.max_rounds,
        verbose=not args.quiet
    )
    
    if args.json:
        results = assistant.research_to_dict(args.query)
        print(json.dumps(results, indent=2))
    else:
        result = assistant.research(args.query)
        print_results(result)


if __name__ == "__main__":
    main()
