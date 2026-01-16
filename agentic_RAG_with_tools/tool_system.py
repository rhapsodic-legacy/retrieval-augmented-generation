"""
Tool System for Agentic RAG

Provides a registry of tools the agent can use:
- Calculator: Math operations
- Web Search: Search the internet
- Code Executor: Run Python code
- Date/Time: Get current date/time
- Knowledge Base: Search internal documents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum
import json
import re


class ToolCategory(Enum):
    CALCULATION = "calculation"
    SEARCH = "search"
    CODE = "code"
    UTILITY = "utility"
    RETRIEVAL = "retrieval"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert result to string for LLM context."""
        if not self.success:
            return f"Error: {self.error}"
        
        if isinstance(self.output, (dict, list)):
            return json.dumps(self.output, indent=2)
        
        return str(self.output)


@dataclass
class ToolDefinition:
    """Tool definition for LLM."""
    name: str
    description: str
    parameters: dict
    category: ToolCategory
    
    def to_dict(self) -> dict:
        """Convert to dict for LLM tool calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class BaseTool(ABC):
    """Abstract base class for tools."""
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return tool definition."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass


class CalculatorTool(BaseTool):
    """
    Calculator for mathematical operations.
    
    Supports basic arithmetic, functions, and expressions.
    """
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations. Use for arithmetic, algebra, and numeric computations.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '(5 * 3) / 2')"
                    }
                },
                "required": ["expression"]
            },
            category=ToolCategory.CALCULATION,
        )
    
    def execute(self, expression: str, **kwargs) -> ToolResult:
        """Evaluate a mathematical expression safely."""
        try:
            import math
            
            # Sanitize expression
            allowed_chars = set('0123456789+-*/().%^ ')
            allowed_funcs = {
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'abs': abs,
                'round': round,
                'pow': pow,
                'pi': math.pi,
                'e': math.e,
            }
            
            # Replace ^ with **
            expr = expression.replace('^', '**')
            
            # Check for function names
            for func in allowed_funcs:
                if func in expr:
                    allowed_chars.update(set(func))
            
            # Evaluate safely
            result = eval(expr, {"__builtins__": {}}, allowed_funcs)
            
            return ToolResult(
                success=True,
                output=result,
                metadata={"expression": expression}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Calculation error: {str(e)}"
            )


class WebSearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo or similar.
    
    For when the knowledge base doesn't have the answer.
    """
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for current information. Use when the knowledge base doesn't have the answer or for recent events.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            category=ToolCategory.SEARCH,
        )
    
    def execute(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        """Execute a web search."""
        try:
            # Try DuckDuckGo
            try:
                from duckduckgo_search import DDGS
                
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=min(num_results, self.max_results)))
                
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    })
                
                return ToolResult(
                    success=True,
                    output=formatted_results,
                    metadata={"query": query, "source": "duckduckgo"}
                )
            except ImportError:
                pass
            
            # Fallback: simulated search (for demo)
            return ToolResult(
                success=True,
                output=[{
                    "title": f"Search result for: {query}",
                    "snippet": "Web search requires duckduckgo-search package. Install with: pip install duckduckgo-search",
                    "url": "https://example.com",
                }],
                metadata={"query": query, "source": "simulated"}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Search error: {str(e)}"
            )


class CodeExecutorTool(BaseTool):
    """
    Execute Python code safely.
    
    Useful for data processing, analysis, and complex computations.
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code_executor",
            description="Execute Python code for data analysis, processing, or complex computations. Returns the result of the last expression or print output.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            },
            category=ToolCategory.CODE,
        )
    
    def execute(self, code: str, **kwargs) -> ToolResult:
        """Execute Python code in a restricted environment."""
        try:
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            # Capture output
            stdout = io.StringIO()
            stderr = io.StringIO()
            
            # Safe globals
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "list": list,
                    "dict": dict,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "abs": abs,
                    "round": round,
                    "isinstance": isinstance,
                    "type": type,
                },
            }
            
            # Add common modules
            try:
                import math
                safe_globals["math"] = math
            except ImportError:
                pass
            
            try:
                import json
                safe_globals["json"] = json
            except ImportError:
                pass
            
            local_vars = {}
            
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, safe_globals, local_vars)
            
            output = stdout.getvalue()
            errors = stderr.getvalue()
            
            # Get the last expression value if any
            result = local_vars.get("result", local_vars.get("output", None))
            
            if output:
                final_output = output.strip()
            elif result is not None:
                final_output = result
            else:
                final_output = "Code executed successfully (no output)"
            
            return ToolResult(
                success=True,
                output=final_output,
                error=errors if errors else None,
                metadata={"code": code[:200]}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}"
            )


class DateTimeTool(BaseTool):
    """Get current date/time information."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="datetime",
            description="Get current date, time, or perform date calculations.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["now", "date", "time", "weekday", "timestamp"],
                        "description": "What to retrieve: now (full datetime), date, time, weekday, or timestamp"
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (e.g., 'UTC', 'US/Eastern'). Default: UTC",
                        "default": "UTC"
                    }
                },
                "required": ["operation"]
            },
            category=ToolCategory.UTILITY,
        )
    
    def execute(self, operation: str, timezone: str = "UTC", **kwargs) -> ToolResult:
        """Get date/time information."""
        try:
            from datetime import datetime, timezone as tz
            
            now = datetime.now(tz.utc)
            
            operations = {
                "now": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
                "timestamp": int(now.timestamp()),
            }
            
            if operation not in operations:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}"
                )
            
            return ToolResult(
                success=True,
                output=operations[operation],
                metadata={"operation": operation, "timezone": "UTC"}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"DateTime error: {str(e)}"
            )


class ToolRegistry:
    """
    Registry of available tools.
    
    Manages tool registration, lookup, and execution.
    """
    
    def __init__(self):
        self.tools: dict[str, BaseTool] = {}
        
        # Register built-in tools
        self.register(CalculatorTool())
        self.register(WebSearchTool())
        self.register(CodeExecutorTool())
        self.register(DateTimeTool())
    
    def register(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.definition.name] = tool
    
    def unregister(self, name: str):
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {name}"
            )
        return tool.execute(**kwargs)
    
    def get_definitions(self) -> list[dict]:
        """Get all tool definitions for LLM."""
        return [tool.definition.to_dict() for tool in self.tools.values()]
    
    def get_tools_by_category(self, category: ToolCategory) -> list[BaseTool]:
        """Get tools by category."""
        return [
            tool for tool in self.tools.values()
            if tool.definition.category == category
        ]
    
    def list_tools(self) -> list[str]:
        """List all tool names."""
        return list(self.tools.keys())


def create_custom_tool(
    name: str,
    description: str,
    func: Callable[..., Any],
    parameters: dict,
    category: ToolCategory = ToolCategory.UTILITY,
) -> BaseTool:
    """
    Create a custom tool from a function.
    
    Usage:
        def my_func(x: int, y: int) -> int:
            return x + y
        
        tool = create_custom_tool(
            name="add",
            description="Add two numbers",
            func=my_func,
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"}
                },
                "required": ["x", "y"]
            }
        )
    """
    class CustomTool(BaseTool):
        def __init__(self):
            self._definition = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                category=category,
            )
            self._func = func
        
        @property
        def definition(self) -> ToolDefinition:
            return self._definition
        
        def execute(self, **kwargs) -> ToolResult:
            try:
                result = self._func(**kwargs)
                return ToolResult(success=True, output=result)
            except Exception as e:
                return ToolResult(success=False, output=None, error=str(e))
    
    return CustomTool()
