#!/usr/bin/env python3
"""
Hierarchical Document RAG with AutoGen - Command Line Interface

Usage:
    # Index and query with group chat (full multi-agent)
    python main.py --workflow group_chat --file docs/report.md query "What are the key findings?"
    
    # Interactive mode with sequential workflow
    python main.py --workflow sequential interactive --directory ./docs
    
    # Simple two-agent setup
    python main.py --workflow simple --file doc.md query "Summarize section 3"
"""

import argparse
import sys
from pathlib import Path

from hierarchical_rag_autogen import HierarchicalRAGAutoGen, RAGConfig


def run_query(args):
    """Run a single query."""
    config = RAGConfig(
        model=args.model,
        workflow_type=args.workflow,
        verbose=not args.quiet,
    )
    
    rag = HierarchicalRAGAutoGen(config)
    
    # Add documents
    if args.file:
        for f in args.file:
            rag.add_document(f)
    elif args.directory:
        rag.add_documents_from_directory(args.directory)
    else:
        print("Error: Specify --file or --directory")
        sys.exit(1)
    
    # Build index
    rag.build_index()
    
    # Query
    question = ' '.join(args.question)
    
    if args.details:
        result = rag.query_with_details(question)
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result["answer"])
        
        if "conversation" in result:
            print("\n" + "-"*60)
            print("AGENT CONVERSATION:")
            print("-"*60)
            for msg in result["conversation"]:
                print(f"\n[{msg.get('name', 'Unknown')}]")
                print(msg.get('content', '')[:500])
    else:
        answer = rag.query(question)
        print("\n" + "="*60)
        print(answer)
        print("="*60)


def run_interactive(args):
    """Run interactive query mode."""
    print("\n" + "="*60)
    print("🤖 Hierarchical RAG with AutoGen - Interactive Mode")
    print(f"   Workflow: {args.workflow}")
    print("="*60)
    
    config = RAGConfig(
        model=args.model,
        workflow_type=args.workflow,
        verbose=True,
    )
    
    rag = HierarchicalRAGAutoGen(config)
    
    # Add documents
    if args.file:
        for f in args.file:
            rag.add_document(f)
    elif args.directory:
        rag.add_documents_from_directory(args.directory)
    else:
        print("\nNo documents specified. Enter file paths (empty line to finish):")
        while True:
            path = input("  File: ").strip()
            if not path:
                break
            if Path(path).exists():
                rag.add_document(path)
            else:
                print(f"    Not found: {path}")
    
    if not rag.parser.documents:
        print("No documents added. Exiting.")
        return
    
    # Build index
    rag.build_index()
    
    print("\n" + "-"*60)
    print("Ready! Type 'quit' to exit, 'docs' to list documents")
    print("-"*60)
    
    while True:
        try:
            question = input("\n🔍 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        if question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if question.lower() == 'docs':
            for doc_id, doc in rag.parser.documents.items():
                print(f"  📄 {doc.title} ({doc.total_sections} sections)")
            continue
        
        try:
            answer = rag.query(question)
            print("\n" + "-"*60)
            print(answer)
            print("-"*60)
        except Exception as e:
            print(f"Error: {e}")


def run_demo(args):
    """Run a demo with sample document."""
    print("\n" + "="*60)
    print("🤖 Hierarchical RAG with AutoGen - DEMO")
    print("="*60)
    
    # Check for example document
    example_path = Path(__file__).parent / "examples" / "software_best_practices.md"
    
    if not example_path.exists():
        print(f"\nExample document not found: {example_path}")
        print("Creating sample document...")
        
        example_path.parent.mkdir(parents=True, exist_ok=True)
        example_path.write_text("""# Sample Document

## Section 1: Introduction
This is an introduction paragraph with some content.

## Section 2: Details
This section contains more detailed information.
See Section 1 for background.

### Section 2.1: Specifics
Very specific details here. As mentioned in Section 2, we need context.
""")
    
    print(f"\n📄 Using: {example_path}")
    
    config = RAGConfig(
        model=args.model,
        workflow_type=args.workflow,
        verbose=True,
    )
    
    rag = HierarchicalRAGAutoGen(config)
    rag.add_document(str(example_path))
    rag.build_index()
    
    # Demo queries
    queries = [
        "What is covered in Section 2?",
        "Give me an overview of the document structure.",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print('='*60)
        
        answer = rag.query(query)
        print(f"\n{answer}")


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Document RAG with AutoGen Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow types:
  group_chat  - Full multi-agent collaboration (most capable)
  sequential  - Fixed agent sequence (more predictable)
  simple      - Two-agent setup (fastest)

Examples:
  python main.py --file doc.md query "What is the summary?"
  python main.py --workflow group_chat --directory ./docs interactive
  python main.py demo
"""
    )
    
    # Global options
    parser.add_argument(
        "--workflow", "-w",
        choices=["group_chat", "sequential", "simple"],
        default="simple",
        help="Agent workflow type (default: simple)"
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-20250514",
        help="LLM model to use"
    )
    parser.add_argument(
        "--file", "-f",
        nargs="+",
        help="Document files to load"
    )
    parser.add_argument(
        "--directory", "-d",
        help="Directory of documents to load"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", nargs="+", help="The question to ask")
    query_parser.add_argument("--details", action="store_true", help="Show full agent conversation")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Interactive query mode")
    
    # Demo command
    subparsers.add_parser("demo", help="Run demo with sample document")
    
    args = parser.parse_args()
    
    if args.command == "query":
        run_query(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
