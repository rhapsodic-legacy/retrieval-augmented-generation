#!/usr/bin/env python3
"""
Agentic RAG - Command Line Interface

Usage:
    python main.py serve                    # Start web server
    python main.py interactive              # Interactive CLI mode
    python main.py query "What is 2+2?"     # Single query
    python main.py add file.txt             # Add document
"""

import argparse
import sys
import os
from typing import Optional

# Rich console for pretty output (fallback to basic if not available)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_header():
    """Print the CLI header."""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold blue]🤖 Agentic RAG[/bold blue]\n"
            "[dim]Smart Assistant with Tool Use[/dim]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*50)
        print("🤖 Agentic RAG - Smart Assistant with Tool Use")
        print("="*50 + "\n")


def print_step(step_type: str, description: str, result: Optional[str] = None):
    """Print an agent step."""
    icons = {
        "routing": "🔀",
        "retrieval": "📚",
        "tool": "🔧",
        "generation": "✨",
    }
    icon = icons.get(step_type, "▶")
    
    if RICH_AVAILABLE:
        color = {
            "routing": "purple",
            "retrieval": "blue",
            "tool": "green",
            "generation": "yellow",
        }.get(step_type, "white")
        
        console.print(f"  [{color}]{icon} {description}[/{color}]")
        if result:
            console.print(f"    [dim]{result}[/dim]")
    else:
        print(f"  {icon} {description}")
        if result:
            print(f"     → {result}")


def print_response(response, verbose: bool = False):
    """Print a response."""
    if RICH_AVAILABLE:
        # Route badge
        route_colors = {
            "direct": "purple",
            "retrieval": "blue",
            "tool_use": "green",
            "web_search": "yellow",
            "hybrid": "magenta",
        }
        route = response.route_decision.route_type.value if response.route_decision else "unknown"
        color = route_colors.get(route, "white")
        
        console.print(f"\n[bold {color}]Route: {route.upper()}[/bold {color}] - {response.route_decision.reasoning if response.route_decision else ''}")
        
        # Steps
        if verbose and response.steps:
            console.print("\n[bold]Steps:[/bold]")
            for step in response.steps:
                print_step(step.step_type, step.description, step.result)
        
        # Tool results
        if response.tool_results:
            console.print("\n[bold green]Tool Results:[/bold green]")
            for tool_name, result in response.tool_results:
                console.print(f"  🔧 {tool_name}: {result.to_string()}")
        
        # Answer
        console.print("\n[bold]Answer:[/bold]")
        console.print(Panel(response.answer, border_style="green"))
        
        # Confidence
        conf_color = "green" if response.confidence >= 0.7 else "yellow" if response.confidence >= 0.4 else "red"
        console.print(f"\n[{conf_color}]Confidence: {response.confidence:.0%}[/{conf_color}]")
        
        # Citations
        if response.citations:
            console.print("\n[bold]Sources:[/bold]")
            for i, cite in enumerate(response.citations, 1):
                console.print(f"  [{i}] {cite.source_name} ({cite.source_type.value}) - {cite.confidence:.0%}")
        
        # Time
        console.print(f"\n[dim]Time: {response.total_time_ms}ms | Model: {response.model_used}[/dim]")
    else:
        # Plain text output
        route = response.route_decision.route_type.value if response.route_decision else "unknown"
        print(f"\nRoute: {route.upper()}")
        print(f"Reasoning: {response.route_decision.reasoning if response.route_decision else 'N/A'}")
        
        if verbose and response.steps:
            print("\nSteps:")
            for step in response.steps:
                print_step(step.step_type, step.description, step.result)
        
        if response.tool_results:
            print("\nTool Results:")
            for tool_name, result in response.tool_results:
                print(f"  {tool_name}: {result.to_string()}")
        
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nConfidence: {response.confidence:.0%}")
        
        if response.citations:
            print("\nSources:")
            for i, cite in enumerate(response.citations, 1):
                print(f"  [{i}] {cite.source_name} ({cite.source_type.value})")
        
        print(f"\nTime: {response.total_time_ms}ms | Model: {response.model_used}")


def run_serve(args):
    """Run the web server."""
    import uvicorn
    
    print_header()
    
    print(f"🌐 Starting server at http://localhost:{args.port}")
    print(f"📖 API docs at http://localhost:{args.port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def run_interactive(args):
    """Run interactive CLI mode."""
    from agentic_rag import AgenticRAG, AgenticRAGConfig
    
    print_header()
    
    # Initialize
    config = AgenticRAGConfig(
        llm_provider=args.provider,
        embedding_provider=args.embedding,
        enable_tools=True,
        enable_web_search=True,
        enable_reflection=True,
    )
    
    print("⚙️  Initializing...")
    rag = AgenticRAG(config)
    
    # Set up step callback for verbose mode
    if args.verbose:
        rag.on_step = lambda step: print_step(step.step_type, step.description, step.result)
    
    # Add files if specified
    if args.files:
        for file_path in args.files:
            if os.path.exists(file_path):
                print(f"📄 Adding {file_path}...")
                rag.add_file(file_path)
    
    print("\n" + "-"*50)
    print("Commands:")
    print("  /quit           - Exit")
    print("  /add FILE       - Add a document")
    print("  /tools          - List tools")
    print("  /stats          - Show statistics")
    print("  /verbose        - Toggle verbose mode")
    print("-"*50 + "\n")
    
    verbose = args.verbose
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold blue]You:[/bold blue] ")
            else:
                user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break
        
        user_input = user_input.strip()
        
        if not user_input:
            continue
        
        # Commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye! 👋")
                break
            
            elif cmd == "/add" and arg:
                if os.path.exists(arg):
                    rag.add_file(arg)
                    print(f"✅ Added {arg}")
                else:
                    print(f"❌ File not found: {arg}")
                continue
            
            elif cmd == "/tools":
                tools = rag.list_tools()
                print("\n🔧 Available Tools:")
                for t in tools:
                    print(f"  - {t}")
                print()
                continue
            
            elif cmd == "/stats":
                stats = rag.get_stats()
                print(f"\n📊 Statistics:")
                print(f"  Knowledge Base: {stats['knowledge_base']['count']} documents")
                print(f"  Tools: {', '.join(stats['tools'])}")
                print()
                continue
            
            elif cmd == "/verbose":
                verbose = not verbose
                if verbose:
                    rag.on_step = lambda step: print_step(step.step_type, step.description, step.result)
                else:
                    rag.on_step = None
                print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                continue
            
            else:
                print(f"Unknown command: {cmd}")
                continue
        
        # Query
        if verbose:
            print("\n🤔 Thinking...\n")
        
        response = rag.query(user_input)
        print_response(response, verbose)
        print()


def run_query(args):
    """Run a single query."""
    from agentic_rag import AgenticRAG, AgenticRAGConfig
    
    config = AgenticRAGConfig(
        llm_provider=args.provider,
        embedding_provider=args.embedding,
    )
    
    rag = AgenticRAG(config)
    
    # Add files if specified
    if args.files:
        for file_path in args.files:
            if os.path.exists(file_path):
                rag.add_file(file_path)
    
    question = ' '.join(args.question)
    
    if args.verbose:
        rag.on_step = lambda step: print_step(step.step_type, step.description, step.result)
        print("🤔 Thinking...\n")
    
    response = rag.query(question)
    print_response(response, args.verbose)


def run_add(args):
    """Add a document to the knowledge base."""
    from agentic_rag import AgenticRAG, AgenticRAGConfig
    
    config = AgenticRAGConfig(
        llm_provider=args.provider,
        embedding_provider=args.embedding,
    )
    
    rag = AgenticRAG(config)
    
    for file_path in args.files:
        if os.path.exists(file_path):
            ids = rag.add_file(file_path)
            print(f"✅ Added {file_path} ({len(ids)} chunks)")
        else:
            print(f"❌ File not found: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG - Smart Assistant with Tool Use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["gemini", "anthropic", "openai"],
        default="gemini",
        help="LLM provider (default: gemini)"
    )
    parser.add_argument(
        "--embedding", "-e",
        choices=["local", "voyage", "openai"],
        default="local",
        help="Embedding provider (default: local)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed reasoning steps"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    # Interactive
    interactive_parser = subparsers.add_parser("interactive", help="Interactive CLI mode")
    interactive_parser.add_argument("files", nargs="*", help="Files to add to knowledge base")
    
    # Query
    query_parser = subparsers.add_parser("query", help="Single query")
    query_parser.add_argument("question", nargs="+", help="Question to ask")
    query_parser.add_argument("-f", "--files", nargs="*", default=[], help="Files to add")
    
    # Add
    add_parser = subparsers.add_parser("add", help="Add documents")
    add_parser.add_argument("files", nargs="+", help="Files to add")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        run_serve(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "add":
        run_add(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
