import ast
import builtins
import os

from pydantic_core.core_schema import TaggedUnionSchema
class DependencyExtractor:
    def __init__(self, source_code):
        self.tree = ast.parse(source_code)
        self.source_lines = source_code.splitlines()
        
        # Maps to store definitions
        self.definitions = {} # {name: ast_node}
        self.imports = []     # List of import nodes
        self.constants = {}   # Global assignments
        
        # Build the index of the file
        self._index_file()
        
        self.collected_definitions = set()
        self.collected_imports = set()
        self.visited = set()

    def _index_file(self):
        """Scans the file to locate all functions, classes, and imports."""
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.definitions[node.name] = node
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imports.append(node)
            elif isinstance(node, ast.Assign):
                # heuristic for global constants
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.constants[target.id] = node

    def get_dependencies(self, target_name):
        """Recursively finds dependencies for a given function/class name."""
        if target_name in self.visited:
            return
        self.visited.add(target_name)

        # 1. Check if it's a defined function/class
        if target_name in self.definitions:
            self.collected_definitions.add(target_name)
            node = self.definitions[target_name]
            self._scan_node_for_usage(node)
        
        # 2. Check if it's a global constant/variable
        elif target_name in self.constants:
            # We treat constants as definitions to extract
            self.collected_definitions.add(target_name) 
            # Constants usually don't have further dependencies, but we could scan them
            
        # 3. Check if it's an import (Basic string matching on imports)
        # AST import handling is tricky, so we scan imports to see if they provide this name
        else:
            self._check_imports(target_name)

    def _scan_node_for_usage(self, node):
        """Scans a function/class body to see what OTHER names it uses."""
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                name = child.id
                if name not in dir(builtins): # Ignore print, len, etc.
                    self.get_dependencies(name)
            # Handle decorators
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                for decorator in child.decorator_list:
                    if isinstance(decorator, ast.Name):
                        self.get_dependencies(decorator.id)
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        self.get_dependencies(decorator.func.id)

    def _check_imports(self, name):
        """Checks if a name comes from an import statement."""
        for node in self.imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname == name or alias.name == name:
                        self.collected_imports.add(node)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == name or alias.asname == name:
                        self.collected_imports.add(node)
                    # Handle 'from module import *' - very hard to trace statically
                    # so we act conservatively:
                    if alias.name == '*': 
                        self.collected_imports.add(node)

    def generate_code(self):
        """Reconstructs the code from collected nodes."""
        output = []
        
        # 1. Add Imports
        # Sort by line number to keep original order roughly
        sorted_imports = sorted(list(self.collected_imports), key=lambda x: x.lineno)
        for node in sorted_imports:
            output.append(ast.get_source_segment("\n".join(self.source_lines), node))
        
        output.append("\n\n# --- Extracted Dependencies ---\n")

        # 2. Add Definitions (Classes/Functions)
        # We try to respect file order to avoid "defined before used" issues
        sorted_defs = sorted(
            [self.definitions[k] for k in self.collected_definitions if k in self.definitions] +
            [self.constants[k] for k in self.collected_definitions if k in self.constants],
            key=lambda x: x.lineno
        )

        for node in sorted_defs:
            segment = ast.get_source_segment("\n".join(self.source_lines), node)
            output.append(segment)
            output.append("") # Add spacing

        return "\n".join(output)

