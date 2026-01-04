#!/usr/bin/env python3
"""
Hierarchical Document RAG - Command Line Interface

A production-grade RAG system with document structure understanding.

Usage:
    # Index documents
    python main.py index --directory ./docs
    python main.py index --file document.md
    
    # Query
    python main.py query "What is the main argument?"
    
    # Interactive mode
    python main.py interactive
    
    # Show document structure
    python main.py tree --file document.md
"""

import argparse
import sys
from pathlib import Path

from hierarchical_rag import HierarchicalRAG, DocumentParser


def cmd_index(args):
    """Index documents into the RAG system."""
    rag = HierarchicalRAG(
        persist_directory=args.persist_dir,
        auto_summarize=not args.no_summaries,
        verbose=True
    )
    
    if args.directory:
        print(f"\n📁 Indexing directory: {args.directory}")
        rag.add_documents_from_directory(
            args.directory,
            extensions=args.extensions.split(',')
        )
    elif args.file:
        for f in args.file:
            rag.add_document(f)
    else:
        print("Error: Specify --directory or --file")
        sys.exit(1)
    
    rag.build_index()
    
    if args.export:
        rag.export_index(args.export)
    
    print("\n✅ Indexing complete!")


def cmd_query(args):
    """Query the RAG system."""
    rag = HierarchicalRAG(
        persist_directory=args.persist_dir,
        verbose=False
    )
    
    # Check if we have indexed data
    stats = rag.vector_store.get_stats()
    if stats['total_content_nodes'] == 0:
        print("Error: No indexed documents. Run 'index' command first.")
        sys.exit(1)
    
    # Need to rebuild retriever from persisted data
    # (In a real implementation, you'd load the nodes from persistence too)
    print("Note: Using persisted vectors. For full functionality, re-index documents.")
    
    question = ' '.join(args.question)
    
    if args.drill_down:
        result = rag.query_with_drill_down(question, args.broad_query)
    else:
        result = rag.query(question, n_results=args.n_results)
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    print(result.answer)
    print(f"\n{'='*60}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Sources used: {len(result.retrieval_results)}")


def cmd_interactive(args):
    """Interactive query mode."""
    print("\n" + "="*60)
    print("🔬 Hierarchical Document RAG - Interactive Mode")
    print("="*60)
    
    # First, we need documents
    if args.directory:
        rag = HierarchicalRAG(
            persist_directory=args.persist_dir,
            auto_summarize=True,
            verbose=True
        )
        rag.add_documents_from_directory(args.directory)
        rag.build_index()
    elif args.file:
        rag = HierarchicalRAG(
            persist_directory=args.persist_dir,
            auto_summarize=True,
            verbose=True
        )
        for f in args.file:
            rag.add_document(f)
        rag.build_index()
    else:
        print("\nNo documents specified. Enter documents to index:")
        print("  (Enter file paths one per line, empty line to finish)")
        
        rag = HierarchicalRAG(auto_summarize=True, verbose=True)
        
        while True:
            path = input("File path: ").strip()
            if not path:
                break
            if Path(path).exists():
                rag.add_document(path)
            else:
                print(f"  File not found: {path}")
        
        if not rag.parser.documents:
            print("No documents added. Exiting.")
            sys.exit(0)
        
        rag.build_index()
    
    print("\n" + "-"*60)
    print("Ready for queries! (Type 'quit' to exit, 'tree' to show structure)")
    print("-"*60 + "\n")
    
    while True:
        try:
            question = input("🔍 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        if question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if question.lower() == 'tree':
            for doc_id in rag.parser.documents:
                tree = rag.get_section_tree(doc_id)
                print_tree(tree)
            continue
        
        if question.lower().startswith('summary'):
            for doc_id, doc in rag.parser.documents.items():
                print(f"\n📄 {doc.title}")
                print(f"   {doc.summary or 'No summary available'}")
            continue
        
        try:
            result = rag.query(question)
            
            print(f"\n{'─'*60}")
            print(result.answer)
            print(f"{'─'*60}")
            print(f"📊 Confidence: {result.confidence:.2f} | Sources: {len(result.retrieval_results)}")
            print()
            
        except Exception as e:
            print(f"Error: {e}")


def cmd_tree(args):
    """Show document structure."""
    parser = DocumentParser()
    doc = parser.parse_file(args.file)
    
    print(f"\n📄 Document Structure: {doc.title}")
    print("="*60)
    
    def print_node(node_id, indent=0):
        node = parser.nodes.get(node_id)
        if not node:
            return
        
        prefix = "  " * indent
        icon = {"document": "📄", "section": "📑", "paragraph": "📝"}.get(
            node.node_type.value, "•"
        )
        
        title = node.title[:50] + "..." if len(node.title) > 50 else node.title
        print(f"{prefix}{icon} {title}")
        
        for child_id in node.children_ids:
            print_node(child_id, indent + 1)
    
    print_node(doc.id)
    print()


def print_tree(tree, indent=0):
    """Pretty print a tree structure."""
    if not tree:
        return
    
    prefix = "  " * indent
    icon = {"document": "📄", "section": "📑", "paragraph": "📝"}.get(
        tree.get("type", ""), "•"
    )
    
    print(f"{prefix}{icon} {tree.get('title', 'Untitled')}")
    
    if tree.get("summary"):
        print(f"{prefix}   └─ {tree['summary']}")
    
    for child in tree.get("children", []):
        print_tree(child, indent + 1)


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Document RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a directory of markdown files
  python main.py index --directory ./docs --persist-dir ./index
  
  # Query the indexed documents
  python main.py query "What are the main findings?" --persist-dir ./index
  
  # Interactive mode with a specific document
  python main.py interactive --file report.md
  
  # Show document structure
  python main.py tree --file document.md
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--directory", "-d", help="Directory of documents")
    index_parser.add_argument("--file", "-f", nargs="+", help="Specific files to index")
    index_parser.add_argument("--persist-dir", "-p", help="Directory to persist index")
    index_parser.add_argument("--extensions", default=".md,.txt", help="File extensions (comma-separated)")
    index_parser.add_argument("--no-summaries", action="store_true", help="Skip summary generation")
    index_parser.add_argument("--export", "-e", help="Export index metadata to JSON")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", nargs="+", help="Question to ask")
    query_parser.add_argument("--persist-dir", "-p", help="Directory with persisted index")
    query_parser.add_argument("--n-results", "-n", type=int, default=5, help="Number of results")
    query_parser.add_argument("--drill-down", action="store_true", help="Use drill-down strategy")
    query_parser.add_argument("--broad-query", help="Broad query for drill-down")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive query mode")
    interactive_parser.add_argument("--directory", "-d", help="Directory of documents")
    interactive_parser.add_argument("--file", "-f", nargs="+", help="Specific files")
    interactive_parser.add_argument("--persist-dir", "-p", help="Directory to persist index")
    
    # Tree command
    tree_parser = subparsers.add_parser("tree", help="Show document structure")
    tree_parser.add_argument("--file", "-f", required=True, help="Document file")
    
    args = parser.parse_args()
    
    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "tree":
        cmd_tree(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
