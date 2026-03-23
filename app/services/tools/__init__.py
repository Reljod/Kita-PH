import importlib
import pkgutil
from typing import Dict, List, Any
from pydantic_ai import FunctionToolset

def get_available_tools() -> Dict[str, str]:
    """
    Scans all modules in this package for tool functions
    and returns a mapping of tool name to its description.
    """
    tools_with_descriptions = {}
    
    for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
        if module_name == "__init__":
            continue
            
        try:
            module = importlib.import_module(f".{module_name}", __package__)
            
            for item_name in dir(module):
                item = getattr(module, item_name)
                # Look for functions that have RunContext in their annotations (likely tools)
                if hasattr(item, "__annotations__") and "RunContext" in str(item.__annotations__.get("ctx", "")):
                    if hasattr(item, "__doc__") and item.__doc__:
                        description = item.__doc__.strip().split("\n")[0]
                        tools_with_descriptions[item_name] = description
        except Exception:
            continue
            
    return tools_with_descriptions

def get_toolsets_by_names(names: List[str]) -> List[FunctionToolset]:
    """
    Returns a list of FunctionToolset objects that contain the tools with the given names.
    """
    found_toolsets = []
    seen_toolsets = set()
    
    for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
        if module_name == "__init__":
            continue
            
        try:
            module = importlib.import_module(f".{module_name}", __package__)
            
            # Find which tools from the list are in this module
            module_tools = []
            for item_name in dir(module):
                if item_name in names:
                    module_tools.append(item_name)
            
            if module_tools:
                # Find the FunctionToolset in this module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, FunctionToolset):
                        if attr not in seen_toolsets:
                            found_toolsets.append(attr)
                            seen_toolsets.add(attr)
        except Exception:
            continue
            
    return found_toolsets
