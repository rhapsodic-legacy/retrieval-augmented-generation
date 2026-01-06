#!/usr/bin/env python3
"""
Hybrid Search RAG - Command Line Interface

Usage:
    # Start API server with frontend
    python main.py serve
    
    # Interactive CLI mode
    python main.py --provider anthropic interactive --file docs/
    
    # Single query
    python main.py query "What is machine learning?" --file doc.md
    
    # Demo mode
    python main.py demo
    
    # Compare search methods
    python main.py compare "search query" --file doc.md
"""

import argparse
import sys
from pathlib import Path


def run_serve(args):
    """Run the FastAPI server with frontend."""
    import uvicorn
    
    print("\n" + "="*60)
    print("🔍 Hybrid Search RAG Server")
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
    from hybrid_rag import HybridRAG, RAGConfig
    
    print("\n" + "="*60)
    print("🔍 Hybrid Search RAG - Interactive Mode")
    print(f"   Provider: {args.provider}")
    print(f"   Fusion: {args.fusion}")
    print("="*60)
    
    # Build config
    config = RAGConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        embedding_provider=args.embedding,
        fusion_strategy=args.fusion,
        enable_vector=not args.no_vector,
        enable_sparse=not args.no_sparse,
        enable_graph=not args.no_graph,
    )
    
    print("\n⚙️  Initializing...")
    rag = HybridRAG(config)
    
    # Load documents
    if args.file:
        for f in args.file:
            path = Path(f)
            if path.is_file():
                print(f"📄 Adding: {f}")
                content = path.read_text(encoding='utf-8')
                rag.add_document(path.stem, content, title=path.name)
            elif path.is_dir():
                for doc_path in path.glob("**/*.md"):
                    print(f"📄 Adding: {doc_path}")
                    content = doc_path.read_text(encoding='utf-8')
                    rag.add_document(doc_path.stem, content, title=doc_path.name)
                for doc_path in path.glob("**/*.txt"):
                    print(f"📄 Adding: {doc_path}")
                    content = doc_path.read_text(encoding='utf-8')
                    rag.add_document(doc_path.stem, content, title=doc_path.name)
    
    stats = rag.get_stats()
    print(f"\n✅ Loaded {stats['documents']} documents")
    
    print("\n" + "-"*60)
    print("Commands:")
    print("  /quit     - Exit")
    print("  /stats    - Show statistics")
    print("  /compare  - Compare search methods")
    print("  /graph    - Show graph stats")
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
            cmd = user_input.lower()
            
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            
            elif cmd == "/stats":
                stats = rag.get_stats()
                print(f"\n📊 Statistics:")
                print(f"   Documents: {stats['documents']}")
                print(f"   Vector store: {stats['search']['vector_store']}")
                print(f"   Sparse index: {stats['search']['sparse_index']}")
                if stats['search']['knowledge_graph']:
                    kg = stats['search']['knowledge_graph']
                    print(f"   Graph: {kg.get('entities', 0)} entities, {kg.get('relations', 0)} relations")
                print()
                continue
            
            elif cmd.startswith("/compare "):
                query = user_input[9:].strip()
                if query:
                    comparison = rag.compare_search_methods(query)
                    print(f"\n🔍 Search Comparison for: '{query}'")
                    print(f"   Vector: {comparison['vector']['count']} results")
                    print(f"   Sparse: {comparison['sparse']['count']} results")
                    print(f"   Overlap: {len(comparison['overlap'])} documents")
                    print(f"   Vector unique: {comparison['vector']['unique'][:3]}")
                    print(f"   Sparse unique: {comparison['sparse']['unique'][:3]}")
                    print()
                continue
            
            elif cmd == "/graph":
                graph_stats = rag.search.knowledge_graph.get_stats()
                print(f"\n🔗 Knowledge Graph:")
                print(f"   Entities: {graph_stats['entities']}")
                print(f"   Relations: {graph_stats['relations']}")
                if graph_stats['entity_types']:
                    print(f"   Entity types: {graph_stats['entity_types']}")
                print()
                continue
            
            else:
                print(f"Unknown command: {cmd}")
                continue
        
        # Regular query
        try:
            response = rag.query(user_input)
            
            print(f"\n📝 Answer:\n{response.answer}\n")
            
            if args.verbose:
                print("📚 Sources:")
                for i, source in enumerate(response.sources[:3], 1):
                    print(f"   [{i}] {source.id} (score: {source.final_score:.4f})")
                    print(f"       Methods: {', '.join(source.sources)}")
                print()
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_query(args):
    """Run a single query."""
    from hybrid_rag import HybridRAG, RAGConfig
    
    config = RAGConfig(
        llm_provider=args.provider,
        embedding_provider=args.embedding,
        fusion_strategy=args.fusion,
    )
    
    rag = HybridRAG(config)
    
    # Load documents
    if args.file:
        for f in args.file:
            path = Path(f)
            if path.is_file():
                content = path.read_text(encoding='utf-8')
                rag.add_document(path.stem, content, title=path.name)
    
    question = ' '.join(args.question)
    response = rag.query(question, k=args.k)
    
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(response.answer)
    print("="*60)
    
    if args.verbose:
        print("\nSOURCES:")
        for i, source in enumerate(response.sources, 1):
            print(f"\n[{i}] {source.id}")
            print(f"    Score: {source.final_score:.4f}")
            print(f"    Methods: {', '.join(source.sources)}")
            if source.vector_score:
                print(f"    Vector: {source.vector_score:.4f}")
            if source.sparse_score:
                print(f"    BM25: {source.sparse_score:.4f}")


def run_compare(args):
    """Compare search methods."""
    from hybrid_rag import HybridRAG, RAGConfig
    
    config = RAGConfig(
        embedding_provider=args.embedding,
    )
    
    rag = HybridRAG(config)
    
    # Load documents
    if args.file:
        for f in args.file:
            path = Path(f)
            if path.is_file():
                content = path.read_text(encoding='utf-8')
                rag.add_document(path.stem, content, title=path.name)
    
    question = ' '.join(args.question)
    comparison = rag.compare_search_methods(question, k=args.k)
    
    print("\n" + "="*60)
    print(f"SEARCH COMPARISON: '{question}'")
    print("="*60)
    
    print(f"\n🎯 VECTOR SEARCH ({comparison['vector']['count']} results):")
    for doc_id in comparison['vector']['ids'][:5]:
        marker = "✓" if doc_id in comparison['overlap'] else " "
        print(f"   {marker} {doc_id}")
    
    print(f"\n📝 BM25 SEARCH ({comparison['sparse']['count']} results):")
    for doc_id in comparison['sparse']['ids'][:5]:
        marker = "✓" if doc_id in comparison['overlap'] else " "
        print(f"   {marker} {doc_id}")
    
    print(f"\n🔄 OVERLAP: {len(comparison['overlap'])} documents")
    print(f"   {comparison['overlap'][:5]}")
    
    print(f"\n🎯 Vector-only: {comparison['vector']['unique'][:3]}")
    print(f"📝 BM25-only: {comparison['sparse']['unique'][:3]}")


def run_demo(args):
    """Run a demo with sample documents."""
    from hybrid_rag import HybridRAG, RAGConfig
    
    print("\n" + "="*60)
    print("🔍 Hybrid Search RAG - Demo")
    print("="*60)
    
    # Sample documents
    docs = [
        {
            "id": "ml_intro",
            "title": "Introduction to Machine Learning",
            "content": """
Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn from data.
There are three main types: supervised learning, unsupervised learning, and reinforcement learning.
Popular algorithms include neural networks, decision trees, and support vector machines (SVMs).
Key applications include image recognition, natural language processing, and recommendation systems.
            """
        },
        {
            "id": "deep_learning",
            "title": "Deep Learning Fundamentals",
            "content": """
Deep Learning uses neural networks with multiple layers to learn hierarchical representations.
Convolutional Neural Networks (CNNs) excel at image processing tasks.
Recurrent Neural Networks (RNNs) and Transformers are used for sequential data like text.
GPT and BERT are famous transformer-based models for natural language processing.
Training requires large datasets and significant computational resources (GPUs/TPUs).
            """
        },
        {
            "id": "nlp_overview",
            "title": "Natural Language Processing",
            "content": """
Natural Language Processing (NLP) enables computers to understand human language.
Key tasks include tokenization, named entity recognition (NER), and sentiment analysis.
Modern NLP relies heavily on transformer architectures like BERT and GPT.
Applications include chatbots, machine translation, and text summarization.
Word embeddings like Word2Vec and GloVe represent words as dense vectors.
            """
        },
        {
            "id": "rag_systems",
            "title": "Retrieval Augmented Generation",
            "content": """
RAG combines information retrieval with language generation for better AI responses.
It works by retrieving relevant documents and using them as context for the LLM.
Hybrid search combines vector similarity (semantic) with keyword matching (BM25).
Knowledge graphs can add relationship information to improve retrieval.
RAG reduces hallucinations by grounding responses in actual documents.
            """
        },
    ]
    
    config = RAGConfig(
        llm_provider=args.provider,
        fusion_strategy="rrf",
    )
    
    print(f"\n⚙️  Using provider: {args.provider}")
    
    rag = HybridRAG(config)
    
    for doc in docs:
        print(f"📄 Adding: {doc['title']}")
        rag.add_document(doc["id"], doc["content"], title=doc["title"])
    
    print(f"\n✅ Added {len(docs)} documents\n")
    
    # Demo queries
    queries = [
        "What is machine learning?",
        "Tell me about transformers and GPT",
        "How does RAG reduce hallucinations?",
    ]
    
    for query in queries:
        print("-"*60)
        print(f"Query: {query}")
        print("-"*60)
        
        response = rag.query(query)
        print(f"\nAnswer: {response.answer}\n")
        
        print("Sources:")
        for source in response.sources[:2]:
            print(f"  • {source.metadata.get('title', source.id)} ({', '.join(source.sources)})")
        print()
    
    # Show comparison
    print("="*60)
    print("SEARCH METHOD COMPARISON")
    print("="*60)
    
    comparison = rag.compare_search_methods("neural networks deep learning")
    print(f"\nQuery: 'neural networks deep learning'")
    print(f"Vector found: {comparison['vector']['ids']}")
    print(f"BM25 found: {comparison['sparse']['ids']}")
    print(f"Overlap: {comparison['overlap']}")
    
    print("\nDemo complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Search RAG - Vector + BM25 + Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start web server with UI
  python main.py serve
  
  # Interactive mode with Claude
  python main.py --provider anthropic interactive --file docs/
  
  # Single query
  python main.py query "What is X?" --file doc.md
  
  # Compare search methods
  python main.py compare "search query" --file doc.md
  
  # Demo mode
  python main.py demo
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
        help="Model name (uses provider default if not specified)"
    )
    parser.add_argument(
        "--embedding", "-e",
        choices=["voyage", "openai", "google", "sentence_transformers", "local"],
        default="local",
        help="Embedding provider (default: local)"
    )
    parser.add_argument(
        "--fusion", "-f",
        choices=["rrf", "weighted", "learned"],
        default="rrf",
        help="Fusion strategy (default: rrf)"
    )
    parser.add_argument(
        "--file",
        nargs="+",
        help="Document files or directories to load"
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Disable vector search"
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable BM25 search"
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable knowledge graph"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results (default: 5)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server with UI")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Interactive CLI mode")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Single query")
    query_parser.add_argument("question", nargs="+", help="Question to ask")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare search methods")
    compare_parser.add_argument("question", nargs="+", help="Search query")
    
    # Demo command
    subparsers.add_parser("demo", help="Run demo with sample documents")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        run_serve(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
