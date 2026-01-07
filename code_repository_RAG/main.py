#!/usr/bin/env python3
"""
Code RAG - Command Line Interface

Usage:
    # Start API server with frontend
    python main.py serve
    
    # Interactive mode
    python main.py --provider anthropic interactive ./my_project
    
    # Single query
    python main.py query "How does authentication work?" ./my_project
    
    # Trace code flow
    python main.py trace login create_session ./my_project
    
    # Find symbol
    python main.py find UserAuth ./my_project
"""

import argparse
import sys
from pathlib import Path


def run_serve(args):
    """Run the FastAPI server."""
    import uvicorn
    
    print("\n" + "="*60)
    print("💻 Code RAG Server")
    print("="*60)
    print(f"\n🌐 Starting server at http://localhost:{args.port}")
    print("📖 API docs at http://localhost:{args.port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def run_interactive(args):
    """Run interactive CLI mode."""
    from code_rag import CodeRAG, CodeRAGConfig
    
    print("\n" + "="*60)
    print("💻 Code RAG - Interactive Mode")
    print(f"   Provider: {args.provider}")
    print("="*60)
    
    config = CodeRAGConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        embedding_provider=args.embedding,
        n_results=args.k,
        include_tests=not args.no_tests,
    )
    
    print("\n⚙️  Initializing...")
    rag = CodeRAG(config)
    
    # Index directory
    if args.directory:
        print(f"\n📂 Indexing: {args.directory}")
        stats = rag.index_directory(args.directory)
        print(f"✅ Indexed {stats['files']} files, {stats['units']} code units")
    
    print("\n" + "-"*60)
    print("Commands:")
    print("  /quit       - Exit")
    print("  /stats      - Show statistics")
    print("  /find NAME  - Find symbol by name")
    print("  /callers NAME - Who calls this function?")
    print("  /trace A B  - Trace flow from A to B")
    print("  /file PATH  - Summarize a file")
    print("-"*60 + "\n")
    
    while True:
        try:
            user_input = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=2)
            cmd = parts[0].lower()
            
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            
            elif cmd == "/stats":
                stats = rag.get_stats()
                print(f"\n📊 Statistics:")
                print(f"   Files indexed: {stats['indexed_files']}")
                print(f"   Total units: {stats['index']['total_units']}")
                print(f"   By type: {stats['index']['by_type']}")
                print(f"   Graph nodes: {stats['graph']['total_nodes']}")
                print(f"   Graph relationships: {stats['graph']['total_relationships']}")
                print()
                continue
            
            elif cmd == "/find" and len(parts) > 1:
                name = parts[1]
                results = rag.find_symbol(name)
                print(f"\n🔍 Found {len(results)} matches for '{name}':")
                for r in results[:10]:
                    print(f"   {r['type']:12} {r['name']:20} {r['file']}:{r['line']}")
                print()
                continue
            
            elif cmd == "/callers" and len(parts) > 1:
                name = parts[1]
                callers = rag.get_callers(name)
                print(f"\n📞 Callers of '{name}':")
                for c in callers[:10]:
                    print(f"   {c['type']:12} {c['name']:20} {c['file']}:{c['line']}")
                print()
                continue
            
            elif cmd == "/trace" and len(parts) > 2:
                start, end = parts[1], parts[2]
                print(f"\n🔄 Tracing: {start} → {end}")
                response = rag.trace_flow(start, end)
                print(f"\n{response.answer}")
                if response.code_flow:
                    print("\n📍 Flow:")
                    for step in response.code_flow:
                        print(f"   {step['step']}. {step['name']} ({step['type']}) in {step['file']}")
                print()
                continue
            
            elif cmd == "/file" and len(parts) > 1:
                file_path = parts[1]
                summary = rag.get_file_summary(file_path)
                print(f"\n📄 {file_path}:")
                if summary.get('classes'):
                    print(f"   Classes: {[c['name'] for c in summary['classes']]}")
                if summary.get('functions'):
                    print(f"   Functions: {[f['name'] for f in summary['functions']]}")
                if summary.get('imports'):
                    print(f"   Imports: {summary['imports'][:5]}")
                print()
                continue
            
            else:
                print(f"Unknown command: {cmd}")
                continue
        
        # Regular query
        try:
            print("\n🔍 Searching...\n")
            response = rag.query(user_input)
            
            print(f"📝 Answer:\n{response.answer}\n")
            
            if args.verbose:
                print("📚 Sources:")
                for i, result in enumerate(response.context.primary_results[:5], 1):
                    unit = result.unit
                    print(f"   [{i}] {unit.type.value}: {unit.name}")
                    print(f"       {unit.file_path}:{unit.start_line}")
                
                if response.context.test_code:
                    print("\n🧪 Related tests:")
                    for test in response.context.test_code[:3]:
                        print(f"   - {test.name} in {test.file_path}")
                print()
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_query(args):
    """Run a single query."""
    from code_rag import CodeRAG, CodeRAGConfig
    
    config = CodeRAGConfig(
        llm_provider=args.provider,
        embedding_provider=args.embedding,
        n_results=args.k,
    )
    
    rag = CodeRAG(config)
    
    # Index
    print(f"Indexing {args.directory}...")
    stats = rag.index_directory(args.directory)
    print(f"Indexed {stats['files']} files, {stats['units']} units")
    
    # Query
    question = ' '.join(args.question)
    print(f"\nQuery: {question}\n")
    
    response = rag.query(question, k=args.k)
    
    print("="*60)
    print("ANSWER:")
    print("="*60)
    print(response.answer)
    print("="*60)
    
    if args.verbose:
        print("\nSOURCES:")
        for i, result in enumerate(response.context.primary_results[:5], 1):
            unit = result.unit
            print(f"\n[{i}] {unit.type.value}: {unit.name}")
            print(f"    File: {unit.file_path}:{unit.start_line}")
            if unit.signature:
                print(f"    Signature: {unit.signature}")


def run_trace(args):
    """Trace code flow."""
    from code_rag import CodeRAG, CodeRAGConfig
    
    config = CodeRAGConfig(
        llm_provider=args.provider,
        embedding_provider=args.embedding,
    )
    
    rag = CodeRAG(config)
    
    # Index
    print(f"Indexing {args.directory}...")
    rag.index_directory(args.directory)
    
    # Trace
    print(f"\nTracing: {args.start} → {args.end}\n")
    
    response = rag.trace_flow(args.start, args.end)
    
    print("="*60)
    print("CODE FLOW:")
    print("="*60)
    print(response.answer)
    
    if response.code_flow:
        print("\n" + "-"*60)
        print("EXECUTION PATH:")
        print("-"*60)
        for step in response.code_flow:
            print(f"{step['step']}. {step['name']}")
            print(f"   Type: {step['type']}")
            print(f"   File: {step['file']}")
            if step.get('signature'):
                print(f"   Signature: {step['signature']}")
            print()


def run_find(args):
    """Find a symbol."""
    from code_rag import CodeRAG, CodeRAGConfig
    
    config = CodeRAGConfig(embedding_provider=args.embedding)
    rag = CodeRAG(config)
    
    # Index
    print(f"Indexing {args.directory}...")
    rag.index_directory(args.directory)
    
    # Find
    name = ' '.join(args.name)
    results = rag.find_symbol(name)
    
    print(f"\n{'='*60}")
    print(f"SYMBOL: {name} ({len(results)} matches)")
    print("="*60)
    
    for r in results:
        print(f"\n{r['type'].upper()}: {r['name']}")
        print(f"  File: {r['file']}:{r['line']}")
        if r.get('signature'):
            print(f"  Signature: {r['signature']}")
        if r.get('docstring'):
            print(f"  Docstring: {r['docstring'][:100]}...")
    
    # Show callers
    if results and args.verbose:
        callers = rag.get_callers(name)
        if callers:
            print(f"\nCALLED BY:")
            for c in callers[:10]:
                print(f"  - {c['name']} ({c['type']}) in {c['file']}")


def main():
    parser = argparse.ArgumentParser(
        description="Code RAG - Intelligent Q&A over codebases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start web server
  python main.py serve
  
  # Interactive mode
  python main.py interactive ./my_project
  
  # Single query
  python main.py query "How does auth work?" ./my_project
  
  # Trace code flow
  python main.py trace login create_session ./my_project
  
  # Find symbol
  python main.py find UserAuth ./my_project
"""
    )
    
    # Global options
    parser.add_argument(
        "--provider", "-p",
        choices=["anthropic", "gemini", "openai"],
        default="anthropic",
        help="LLM provider (default: anthropic)"
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name"
    )
    parser.add_argument(
        "--embedding", "-e",
        choices=["voyage", "openai", "google", "local"],
        default="local",
        help="Embedding provider (default: local)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Exclude test files from results"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    # Interactive
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("directory", help="Directory to index")
    
    # Query
    query_parser = subparsers.add_parser("query", help="Single query")
    query_parser.add_argument("question", nargs="+", help="Question")
    query_parser.add_argument("directory", help="Directory to index")
    
    # Trace
    trace_parser = subparsers.add_parser("trace", help="Trace code flow")
    trace_parser.add_argument("start", help="Starting function")
    trace_parser.add_argument("end", help="Ending function")
    trace_parser.add_argument("directory", help="Directory to index")
    
    # Find
    find_parser = subparsers.add_parser("find", help="Find symbol")
    find_parser.add_argument("name", nargs="+", help="Symbol name")
    find_parser.add_argument("directory", help="Directory to index")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        run_serve(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "trace":
        run_trace(args)
    elif args.command == "find":
        run_find(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
