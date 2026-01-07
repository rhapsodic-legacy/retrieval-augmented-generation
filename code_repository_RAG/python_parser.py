"""
AST-Aware Code Parser for Python

Extracts:
- Functions and methods
- Classes
- Docstrings and comments
- Imports and dependencies
- Decorators
- Type hints
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from enum import Enum


class CodeUnitType(Enum):
    """Types of code units."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    ASYNC_FUNCTION = "async_function"
    PROPERTY = "property"
    CONSTANT = "constant"
    IMPORT = "import"


@dataclass
class CodeUnit:
    """A parsed code unit (function, class, etc.)."""
    id: str
    name: str
    type: CodeUnitType
    file_path: str
    
    # Source code
    source: str
    start_line: int
    end_line: int
    
    # Documentation
    docstring: Optional[str] = None
    comments: list[str] = field(default_factory=list)
    
    # Signature info
    signature: Optional[str] = None
    parameters: list[dict] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    
    # Relationships
    parent_class: Optional[str] = None
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # Functions/methods called
    references: list[str] = field(default_factory=list)  # Variables/names referenced
    
    # Metadata
    is_async: bool = False
    is_private: bool = False
    is_test: bool = False
    complexity: int = 0  # Cyclomatic complexity estimate
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "file_path": self.file_path,
            "source": self.source,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
            "signature": self.signature,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "decorators": self.decorators,
            "parent_class": self.parent_class,
            "imports": self.imports,
            "calls": self.calls,
            "is_async": self.is_async,
            "is_private": self.is_private,
            "is_test": self.is_test,
        }


@dataclass
class ParsedFile:
    """A parsed source file."""
    file_path: str
    language: str
    
    # Module-level info
    module_docstring: Optional[str] = None
    imports: list[dict] = field(default_factory=list)
    
    # Code units
    units: list[CodeUnit] = field(default_factory=list)
    
    # Relationships
    dependencies: list[str] = field(default_factory=list)  # Files this depends on
    
    def get_classes(self) -> list[CodeUnit]:
        return [u for u in self.units if u.type == CodeUnitType.CLASS]
    
    def get_functions(self) -> list[CodeUnit]:
        return [u for u in self.units if u.type in (
            CodeUnitType.FUNCTION,
            CodeUnitType.ASYNC_FUNCTION,
        )]
    
    def get_methods(self) -> list[CodeUnit]:
        return [u for u in self.units if u.type in (
            CodeUnitType.METHOD,
            CodeUnitType.PROPERTY,
        )]


class PythonParser:
    """AST-based parser for Python code."""
    
    def __init__(self):
        self.current_file = ""
        self.source_lines = []
    
    def parse_file(self, file_path: str, content: Optional[str] = None) -> ParsedFile:
        """Parse a Python file."""
        self.current_file = file_path
        
        if content is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        self.source_lines = content.splitlines()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            # Return empty result for unparseable files
            return ParsedFile(
                file_path=file_path,
                language="python",
            )
        
        parsed = ParsedFile(
            file_path=file_path,
            language="python",
        )
        
        # Get module docstring
        parsed.module_docstring = ast.get_docstring(tree)
        
        # Extract imports
        parsed.imports = self._extract_imports(tree)
        parsed.dependencies = [imp["module"] for imp in parsed.imports if imp["module"]]
        
        # Extract code units
        parsed.units = self._extract_units(tree, content)
        
        return parsed
    
    def _extract_imports(self, tree: ast.AST) -> list[dict]:
        """Extract import statements."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "names": None,
                    })
            
            elif isinstance(node, ast.ImportFrom):
                imports.append({
                    "type": "from",
                    "module": node.module,
                    "alias": None,
                    "names": [
                        {"name": alias.name, "alias": alias.asname}
                        for alias in node.names
                    ],
                    "level": node.level,  # For relative imports
                })
        
        return imports
    
    def _extract_units(self, tree: ast.AST, content: str) -> list[CodeUnit]:
        """Extract all code units from the AST."""
        units = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                units.extend(self._parse_class(node, content))
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                units.append(self._parse_function(node, content))
            
            elif isinstance(node, ast.Assign):
                # Module-level constants
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        units.append(self._parse_constant(node, target, content))
        
        return units
    
    def _parse_class(self, node: ast.ClassDef, content: str) -> list[CodeUnit]:
        """Parse a class definition."""
        units = []
        
        class_id = f"{self.current_file}::{node.name}"
        source = self._get_source(node, content)
        
        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_name(base)}")
        
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        class_unit = CodeUnit(
            id=class_id,
            name=node.name,
            type=CodeUnitType.CLASS,
            file_path=self.current_file,
            source=source,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            signature=f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
            decorators=decorators,
            references=bases,
            is_private=node.name.startswith('_'),
            is_test=node.name.startswith('Test') or node.name.endswith('Test'),
        )
        
        # Extract calls from class body
        class_unit.calls = self._extract_calls(node)
        
        units.append(class_unit)
        
        # Parse methods
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._parse_function(child, content, parent_class=node.name)
                units.append(method)
        
        return units
    
    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        content: str,
        parent_class: Optional[str] = None,
    ) -> CodeUnit:
        """Parse a function or method definition."""
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        if parent_class:
            func_id = f"{self.current_file}::{parent_class}.{node.name}"
            unit_type = CodeUnitType.METHOD
            
            # Check for property
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == 'property':
                    unit_type = CodeUnitType.PROPERTY
                    break
        else:
            func_id = f"{self.current_file}::{node.name}"
            unit_type = CodeUnitType.ASYNC_FUNCTION if is_async else CodeUnitType.FUNCTION
        
        source = self._get_source(node, content)
        
        # Parse parameters
        parameters = self._parse_parameters(node.args)
        
        # Get return type
        return_type = None
        if node.returns:
            return_type = self._get_annotation(node.returns)
        
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Build signature
        params_str = ", ".join(
            p.get("name", "") + (f": {p['type']}" if p.get("type") else "")
            for p in parameters
        )
        signature = f"{'async ' if is_async else ''}def {node.name}({params_str})"
        if return_type:
            signature += f" -> {return_type}"
        
        unit = CodeUnit(
            id=func_id,
            name=node.name,
            type=unit_type,
            file_path=self.current_file,
            source=source,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            signature=signature,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            parent_class=parent_class,
            is_async=is_async,
            is_private=node.name.startswith('_') and not node.name.startswith('__'),
            is_test=node.name.startswith('test_') or node.name.startswith('test'),
            complexity=self._estimate_complexity(node),
        )
        
        # Extract function calls
        unit.calls = self._extract_calls(node)
        
        return unit
    
    def _parse_constant(
        self,
        node: ast.Assign,
        target: ast.Name,
        content: str,
    ) -> CodeUnit:
        """Parse a module-level constant."""
        const_id = f"{self.current_file}::{target.id}"
        source = self._get_source(node, content)
        
        return CodeUnit(
            id=const_id,
            name=target.id,
            type=CodeUnitType.CONSTANT,
            file_path=self.current_file,
            source=source,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
        )
    
    def _parse_parameters(self, args: ast.arguments) -> list[dict]:
        """Parse function parameters."""
        params = []
        
        # Regular args
        defaults_offset = len(args.args) - len(args.defaults)
        
        for i, arg in enumerate(args.args):
            param = {
                "name": arg.arg,
                "type": self._get_annotation(arg.annotation) if arg.annotation else None,
            }
            
            # Check for default value
            default_idx = i - defaults_offset
            if default_idx >= 0:
                param["has_default"] = True
            
            params.append(param)
        
        # *args
        if args.vararg:
            params.append({
                "name": f"*{args.vararg.arg}",
                "type": self._get_annotation(args.vararg.annotation) if args.vararg.annotation else None,
            })
        
        # **kwargs
        if args.kwarg:
            params.append({
                "name": f"**{args.kwarg.arg}",
                "type": self._get_annotation(args.kwarg.annotation) if args.kwarg.annotation else None,
            })
        
        return params
    
    def _get_annotation(self, node: ast.AST) -> str:
        """Get type annotation as string."""
        if node is None:
            return None
        
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation(node.value)
            slice_val = self._get_annotation(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Tuple):
            elements = ", ".join(self._get_annotation(e) for e in node.elts)
            return f"({elements})"
        elif isinstance(node, ast.BinOp):
            # Union types: X | Y
            left = self._get_annotation(node.left)
            right = self._get_annotation(node.right)
            return f"{left} | {right}"
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else "..."
    
    def _get_decorator_name(self, node: ast.AST) -> str:
        """Get decorator name as string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else "..."
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.Class')."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return ".".join(reversed(parts))
    
    def _extract_calls(self, node: ast.AST) -> list[str]:
        """Extract function/method calls from a node."""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        
        return list(set(calls))
    
    def _estimate_complexity(self, node: ast.AST) -> int:
        """Estimate cyclomatic complexity."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity
    
    def _get_source(self, node: ast.AST, content: str) -> str:
        """Get source code for a node."""
        if hasattr(ast, 'get_source_segment'):
            source = ast.get_source_segment(content, node)
            if source:
                return source
        
        # Fallback: extract by line numbers
        start = node.lineno - 1
        end = node.end_lineno if node.end_lineno else node.lineno
        return '\n'.join(self.source_lines[start:end])


def parse_python_file(file_path: str, content: Optional[str] = None) -> ParsedFile:
    """Parse a Python file and extract code units."""
    parser = PythonParser()
    return parser.parse_file(file_path, content)
