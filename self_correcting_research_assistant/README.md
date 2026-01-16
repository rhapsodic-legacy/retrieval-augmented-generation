# 🔬 Self-Correcting Research Assistant

An intelligent AI agent that searches the web, cross-references sources, detects contradictions, and automatically re-verifies disputed facts using the Anthropic Claude API. 

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Anthropic](https://img.shields.io/badge/Powered%20by-Claude%20API-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **Intelligent Web Search**: Uses Claude's web search tool to find information from multiple sources
- **Contradiction Detection**: Automatically identifies conflicting claims across different sources  
- **Self-Correction Loop**: Re-runs verification searches when contradictions are found
- **Confidence Scoring**: Rates each claim as high/medium/low confidence based on source agreement
- **Multiple Output Formats**: Pretty-printed CLI output or structured JSON
- **Programmatic API**: Import and use in your own Python projects

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/self-correcting-research-assistant.git
cd self-correcting-research-assistant

# Install dependencies
pip install anthropic

# Set your API key
export ANTHROPIC_API_KEY="your-api-key"
```

### Command Line Usage

```bash
# Basic usage
python self_correcting_research_assistant.py "What is the population of Tokyo?"

# Interactive mode (prompts for query)
python self_correcting_research_assistant.py

# Output as JSON
python self_correcting_research_assistant.py "Latest Mars rover discoveries" --json

# Quiet mode (no progress logs)
python self_correcting_research_assistant.py "Climate change statistics" --quiet

# Custom model and verification rounds
python self_correcting_research_assistant.py "AI regulations in EU" --model claude-opus-4-20250514 --max-rounds 5
```

### Programmatic Usage

```python
from self_correcting_research_assistant import ResearchAssistant

# Create assistant
assistant = ResearchAssistant(
    model="claude-sonnet-4-20250514",
    max_verification_rounds=3,
    verbose=True
)

# Run research
result = assistant.research("What are the health benefits of intermittent fasting?")

# Access structured results
print(f"Found {len(result.findings)} claims")
print(f"Detected {len(result.contradictions)} contradictions")
print(f"Resolved {len(result.resolved_claims)} contradictions")

for finding in result.findings:
    print(f"- {finding.claim} (confidence: {finding.confidence})")

# Or get results as dictionary
results_dict = assistant.research_to_dict("Your query here")
```

## 🎯 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER ENTERS QUERY                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 1: INITIAL RESEARCH                          │
│  • Claude searches the web with web_search tool                 │
│  • Gathers information from multiple sources                    │
│  • Extracts factual claims with confidence levels               │
│  • Identifies contradictions between sources                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Contradictions?  │
                    └─────────┬─────────┘
              No              │              Yes
              │               │               │
              ▼               │               ▼
┌─────────────────────┐       │  ┌─────────────────────────────────┐
│  Return Results     │       │  │    PHASE 2: VERIFICATION        │
│  (Single Iteration) │       │  │  • Focuses on disputed claims   │
└─────────────────────┘       │  │  • Searches authoritative       │
                              │  │    sources                      │
                              │  │  • Resolves contradictions      │
                              │  │  • Repeats if needed            │
                              │  └─────────────────────────────────┘
                              │               │
                              │               ▼
                              │  ┌─────────────────────────────────┐
                              │  │    Return Verified Results      │
                              │  │    (Multiple Iterations)        │
                              │  └─────────────────────────────────┘
                              │               │
                              └───────────────┘
```

## 📖 API Reference

### ResearchAssistant

```python
class ResearchAssistant:
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
```

### Data Classes

```python
@dataclass
class Finding:
    claim: str           # The factual claim
    sources: list[str]   # Source URLs/names
    confidence: str      # "high", "medium", or "low"
    notes: str           # Additional context

@dataclass
class Contradiction:
    topic: str           # Subject of the contradiction
    claim_a: str         # First version of the claim
    source_a: str        # Source for claim A
    claim_b: str         # Contradicting version
    source_b: str        # Source for claim B

@dataclass
class ResolvedClaim:
    original_contradiction: str  # What was disputed
    verified_fact: str          # The accurate information
    explanation: str            # How it was determined
    confidence: str             # Confidence level
    sources: list[str]          # Authoritative sources

@dataclass
class ResearchResult:
    query: str
    findings: list[Finding]
    contradictions: list[Contradiction]
    resolved_claims: list[ResolvedClaim]
    still_uncertain: list[UncertainItem]
    summary: str
    iterations: int
    timestamp: str
```

## 📋 Example Output

```
======================================================================
🔬 SELF-CORRECTING RESEARCH ASSISTANT - RESULTS
======================================================================

📋 Query: What is the tallest building in the world?
🔄 Search Iterations: 2
📊 Claims Found: 4
⚠️  Contradictions Detected: 1
✅ Contradictions Resolved: 1

----------------------------------------------------------------------
📝 SUMMARY
----------------------------------------------------------------------
The Burj Khalifa in Dubai is currently the tallest building in the 
world at 828 meters (2,717 feet), completed in 2010.

----------------------------------------------------------------------
📌 FINDINGS
----------------------------------------------------------------------

1. The Burj Khalifa stands at 828 meters (2,717 feet)
   🟢 Confidence: high
   📎 Sources: Council on Tall Buildings, Wikipedia
   💡 Notes: Measurement includes spire

----------------------------------------------------------------------
✅ VERIFIED RESOLUTIONS
----------------------------------------------------------------------

1. Original Issue: Height measurement methodology
   ✓ Verified Fact: 828m is the official height to architectural top
   🟢 Confidence: high
   📖 Explanation: CTBUH standards confirm measurement methodology
   📎 Sources: CTBUH official database

======================================================================
Research complete!
======================================================================
```

## 🔧 CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `query` | Research topic or question | Interactive prompt |
| `--model` | Claude model to use | `claude-sonnet-4-20250514` |
| `--max-rounds` | Maximum verification rounds | `3` |
| `--json` | Output results as JSON | `False` |
| `--quiet` | Suppress progress logs | `False` |

## 📊 Example Use Cases

- **Fact-checking**: Verify claims from news articles or social media
- **Research synthesis**: Combine information from multiple sources
- **Due diligence**: Cross-reference business or investment information
- **Academic research**: Gather and verify scientific claims
- **Journalism**: Investigate and verify stories

## 🛡️ Limitations

- Results depend on web search availability and quality
- Some contradictions may require human judgment to resolve
- Real-time information may not always be available
- API rate limits apply

## 📁 Project Structure

```
self-correcting-research-assistant/
├── self_correcting_research_assistant.py   # Main module
├── README.md                               # Documentation
├── requirements.txt                        # Dependencies
└── LICENSE                                 # MIT License
```

## 📦 Requirements

```
anthropic>=0.40.0
```

## 📄 License

MIT License - feel free to use, modify, and distribute.

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---


