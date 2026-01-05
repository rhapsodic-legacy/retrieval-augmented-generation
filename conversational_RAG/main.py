#!/usr/bin/env python3
"""
Conversational RAG with Memory - Command Line Interface

Usage:
    # Interactive chat with Claude
    python main.py --provider anthropic interactive --file docs/report.md
    
    # Interactive chat with Gemini
    python main.py --provider gemini interactive --file docs/report.md
    
    # Single query
    python main.py --file doc.md query "What is this about?"
    
    # With persistent memory
    python main.py --storage ./memory --user myuser interactive
"""

import argparse
import sys
from pathlib import Path

from conversational_rag import (
    ConversationalRAG,
    ConversationConfig,
)


def run_interactive(args):
    """Run interactive chat mode."""
    print("\n" + "="*60)
    print("💬 Conversational RAG with Memory")
    print(f"   Provider: {args.provider}")
    print("="*60)
    
    # Build config
    config = ConversationConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        embedding_provider=args.embedding_provider,
        user_id=args.user,
        storage_path=args.storage,
        enable_rewriting=not args.no_rewrite,
    )
    
    # Initialize RAG
    print("\n⚙️  Initializing...")
    rag = ConversationalRAG(config)
    
    # Add documents
    if args.file:
        for f in args.file:
            print(f"📄 Adding: {f}")
            rag.add_document_from_file(f)
    elif args.directory:
        dir_path = Path(args.directory)
        for f in dir_path.glob("**/*.md"):
            print(f"📄 Adding: {f}")
            rag.add_document_from_file(str(f))
        for f in dir_path.glob("**/*.txt"):
            print(f"📄 Adding: {f}")
            rag.add_document_from_file(str(f))
    
    if rag.documents.get_stats()["total_documents"] == 0:
        print("\n⚠️  No documents loaded. Add documents or use --file/--directory")
        print("    You can still chat, but answers will be limited.\n")
    
    print("\n" + "-"*60)
    print("Ready! Commands:")
    print("  /quit     - Exit")
    print("  /new      - Start new session")
    print("  /remember - Remember a fact")
    print("  /stats    - Show statistics")
    print("  /history  - Show conversation history")
    print("-"*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                rag.end_session()
                break
            
            elif cmd == "/new":
                rag.new_session()
                print("🔄 Started new session\n")
                continue
            
            elif cmd.startswith("/remember "):
                fact = user_input[10:].strip()
                rag.remember(fact)
                print(f"✅ Remembered: {fact}\n")
                continue
            
            elif cmd == "/stats":
                stats = rag.get_stats()
                print("\n📊 Statistics:")
                print(f"   Turns: {stats['turn_count']}")
                print(f"   Documents: {stats['documents']['total_documents']}")
                print(f"   Session citations: {stats['citations']['total_citations']}")
                print(f"   Long-term facts: {stats['memory']['long_term']['facts']}")
                print()
                continue
            
            elif cmd == "/history":
                messages = rag.memory.get_conversation_context(10)
                print("\n📜 Recent History:")
                for msg in messages:
                    role = "You" if msg["role"] == "user" else "Assistant"
                    print(f"   {role}: {msg['content'][:100]}...")
                print()
                continue
            
            else:
                print(f"Unknown command: {cmd}\n")
                continue
        
        # Process chat message
        try:
            response = rag.chat(user_input)
            
            # Show rewrite info if applicable
            if response.was_rewritten:
                print(f"   [Understood as: {response.rewritten_query}]")
            
            print(f"\nAssistant: {response.answer}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_query(args):
    """Run a single query."""
    config = ConversationConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        embedding_provider=args.embedding_provider,
    )
    
    rag = ConversationalRAG(config)
    
    # Add documents
    if args.file:
        for f in args.file:
            rag.add_document_from_file(f)
    
    # Run query
    question = ' '.join(args.question)
    response = rag.chat(question)
    
    print("\n" + "="*60)
    print(response.answer)
    print("="*60)
    
    if args.verbose:
        print(f"\nSources used: {len(response.sources_used)}")
        if response.was_rewritten:
            print(f"Query rewritten: {response.rewritten_query}")


def run_demo(args):
    """Run a demo with sample conversation."""
    print("\n" + "="*60)
    print("💬 Conversational RAG Demo")
    print("="*60)
    
    # Create sample document
    sample_doc = """
# Product Requirements Document

## 1. Overview
This document describes the requirements for our new mobile application.
The app will help users track their daily habits and goals.

## 2. Features

### 2.1 User Authentication
Users can sign up with email or social login.
Two-factor authentication is required for premium accounts.

### 2.2 Habit Tracking
Users can create custom habits to track daily.
The app sends reminders at user-specified times.
Progress is visualized with charts and streaks.

### 2.3 Goal Setting
Users can set monthly and yearly goals.
Goals can be linked to specific habits.
The system provides AI-powered suggestions.

## 3. Technical Requirements

### 3.1 Performance
The app must load within 2 seconds on 4G connections.
Offline mode should support basic habit logging.

### 3.2 Security
All data must be encrypted at rest and in transit.
User data retention follows GDPR guidelines.
"""
    
    config = ConversationConfig(
        llm_provider=args.provider,
        enable_rewriting=True,
    )
    
    print(f"\n⚙️  Using provider: {args.provider}")
    
    rag = ConversationalRAG(config)
    rag.add_document(sample_doc, title="Product Requirements", source="PRD.md")
    
    print("📄 Added sample document: Product Requirements\n")
    
    # Demo conversation
    demo_queries = [
        "What is this document about?",
        "Tell me about the authentication features",
        "What about the second feature?",  # Tests pronoun resolution
        "How does it relate to security?",  # Tests context awareness
    ]
    
    for query in demo_queries:
        print(f"You: {query}")
        
        response = rag.chat(query)
        
        if response.was_rewritten:
            print(f"   [Understood as: {response.rewritten_query}]")
        
        print(f"\nAssistant: {response.answer}\n")
        print("-"*40 + "\n")
    
    print("Demo complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Conversational RAG with Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive with Claude
  python main.py --provider anthropic interactive --file doc.md
  
  # Interactive with Gemini
  python main.py --provider gemini interactive --file doc.md
  
  # Single query
  python main.py --file doc.md query "Summarize this document"
  
  # With persistent memory
  python main.py --storage ./memory --user john interactive
  
  # Demo mode
  python main.py demo
"""
    )
    
    # Global options
    parser.add_argument(
        "--provider", "-p",
        choices=["anthropic", "gemini"],
        default="anthropic",
        help="LLM provider (default: anthropic)"
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name (uses provider default if not specified)"
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["voyage", "google", "openai", "local"],
        default="local",
        help="Embedding provider (default: local)"
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
        "--storage", "-s",
        help="Storage path for persistent memory"
    )
    parser.add_argument(
        "--user", "-u",
        default="default",
        help="User ID for memory isolation"
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable query rewriting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Interactive chat mode")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Single query")
    query_parser.add_argument("question", nargs="+", help="Question to ask")
    
    # Demo command
    subparsers.add_parser("demo", help="Run demo")
    
    args = parser.parse_args()
    
    if args.command == "interactive":
        run_interactive(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
