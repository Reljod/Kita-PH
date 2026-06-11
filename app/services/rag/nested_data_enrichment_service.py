import re
from typing import Dict, Any, List, Tuple, Optional, Protocol

def slugify(text: str) -> str:
    """Converts a heading or key to a clean lowercased slug."""
    if not text:
        return ""
    # Strip HTML tags
    text_clean = re.sub(r'<[^>]+>', '', text)
    cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '', text_clean.lower())
    slug = re.sub(r'[\s-]+', '_', cleaned)
    return slug.strip('_')

def set_nested_value(d: dict, path: List[str], key: str, value: Any):
    """Sets a value in a nested dictionary along a path."""
    current = d
    for step in path:
        if step not in current or not isinstance(current[step], dict):
            current[step] = {}
        current = current[step]
    current[key] = value

def get_nested_value(d: dict, path: List[str]) -> Optional[Any]:
    """Retrieves a sub-tree/value in a nested dictionary along a path."""
    current = d
    for step in path:
        if not isinstance(current, dict) or step not in current:
            return None
        current = current[step]
    return current

class INestedDataEnrichmentService(Protocol):
    def build_hierarchy_and_leaves(
        self,
        parse_result: Dict[str, Any], 
        file_id: str, 
        org_id: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        ...

class NestedDataEnrichmentService(INestedDataEnrichmentService):
    def build_hierarchy_and_leaves(
        self,
        parse_result: Dict[str, Any], 
        file_id: str, 
        org_id: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parses the LlamaParse layout output, constructs a clean nested JSON representation,
        and extracts flattened leaves with deterministic metadata.
        """
        nested_json = {}
        leaves = []
        
        # Determine the source of pages
        # First check result["items"]["pages"], then result["pages"], then fallback
        pages = parse_result.get("items", {}).get("pages", [])
        if not pages:
            pages = parse_result.get("pages", [])
            
        if not pages:
            # Fallback if there is a flat result or string in result
            return {}, []
            
        heading_stack: List[Tuple[str, str]] = [] # stack of (slug, original_text)
        counters: Dict[str, int] = {} # paths to index counter for unique naming
        
        for page_obj in pages:
            page_num = page_obj.get("page") or page_obj.get("page_number", 1)
            items = page_obj.get("items", [])
            
            # Fallback if layout items are missing but markdown is present
            if not items and page_obj.get("markdown"):
                items = self._parse_markdown_to_items(page_obj["markdown"])
                
            for item in items:
                item_type = item.get("type", "")
                
                # Check for heading
                is_heading = item_type == "heading"
                item_md = item.get("md", "")
                item_val = item.get("value") or item.get("text") or ""
                
                if not is_heading and item_md.strip().startswith("#"):
                    is_heading = True
                    
                if is_heading:
                    # Get heading text
                    h_text = item_val.strip()
                    if not h_text and item_md:
                        h_text = item_md.lstrip("#").strip()
                    
                    if not h_text:
                        continue
                        
                    # Determine heading level
                    level = item.get("level")
                    if level is None:
                        # count leading '#'
                        level = len(item_md) - len(item_md.lstrip("#"))
                        if level == 0:
                            level = 1
                            
                    slug = slugify(h_text)
                    if not slug:
                        slug = f"section_{level}"
                        
                    # Adjust heading stack to parent level
                    while len(heading_stack) >= level:
                        heading_stack.pop()
                    heading_stack.append((slug, h_text))
                    continue
                
                # If list item, process each sub-item as a leaf
                if item_type == "list" and "items" in item:
                    list_items = item.get("items", [])
                    for sub_item in list_items:
                        sub_text = sub_item.get("value") or sub_item.get("text") or sub_item.get("md") or ""
                        if not sub_text:
                            continue
                        sub_bbox = sub_item.get("bbox") or item.get("bbox") # fallback to parent list bbox
                        self._process_leaf(
                            text_content=sub_text,
                            bbox=sub_bbox,
                            page_num=page_num,
                            heading_stack=heading_stack,
                            counters=counters,
                            nested_json=nested_json,
                            leaves=leaves,
                            file_id=file_id,
                            org_id=org_id,
                            item_type="text"
                        )
                else:
                    # Single text or table leaf
                    if not item_val and item_md:
                        item_val = item_md
                    if not item_val:
                        continue
                    self._process_leaf(
                        text_content=item_val,
                        bbox=item.get("bbox"),
                        page_num=page_num,
                        heading_stack=heading_stack,
                        counters=counters,
                        nested_json=nested_json,
                        leaves=leaves,
                        file_id=file_id,
                        org_id=org_id,
                        item_type=item_type
                    )
                    
        return nested_json, leaves

    def _process_leaf(
        self,
        text_content: str,
        bbox: Optional[Any],
        page_num: int,
        heading_stack: List[Tuple[str, str]],
        counters: Dict[str, int],
        nested_json: dict,
        leaves: list,
        file_id: str,
        org_id: str,
        item_type: str
    ):
        text_content_clean = text_content.strip()
        if not text_content_clean:
            return
            
        # 1. Parse Key-Value (if applicable)
        leaf_key = None
        leaf_val = text_content_clean
        
        if ":" in text_content_clean and not text_content_clean.startswith("http"):
            parts = text_content_clean.split(":", 1)
            k_cand = parts[0].strip()
            v_cand = parts[1].strip()
            # If the candidate key is relatively short and single-line, treat as key
            if 0 < len(k_cand) < 40 and len(v_cand) > 0 and "\n" not in k_cand:
                leaf_key = slugify(k_cand)
                leaf_val = v_cand
                
        # 2. Sequential fallback if not key-value
        heading_path = [slug for slug, _ in heading_stack]
        heading_path_str = ".".join(heading_path)
        
        if not leaf_key:
            idx = counters.get(heading_path_str, 0)
            counters[heading_path_str] = idx + 1
            prefix = "table" if item_type == "table" else "item"
            leaf_key = f"{prefix}_{idx}"
            
        # 3. Store in the dynamic nested tree
        set_nested_value(nested_json, heading_path, leaf_key, leaf_val)
        
        # 4. Form coordinates, breadcrumbs and human readable representations
        json_path = ".".join(heading_path + [leaf_key])
        
        heading_hierarchy = [title for _, title in heading_stack]
        heading_text = " > ".join(heading_hierarchy)
        
        # Breadcrumb can represent the path or full breadcrumb hierarchy
        breadcrumb = json_path
        
        # Text to be embedded/indexed
        indexed_text = f"{json_path}: {leaf_val}"
        
        # Human-readable "heading > text" format
        heading_to_text = f"{heading_text} > {leaf_val}" if heading_text else leaf_val
        
        # Extract location safely (supports list of dict bounding boxes)
        location = None
        if bbox:
            if isinstance(bbox, list) and len(bbox) > 0:
                location = bbox[0]
            elif isinstance(bbox, dict):
                location = bbox
                
        leaves.append({
            "file_id": file_id,
            "org_id": org_id,
            "json_path": json_path,
            "breadcrumb": breadcrumb,
            "heading_text": heading_text,
            "text": indexed_text,
            "heading_to_text": heading_to_text,
            "content": leaf_val,
            "page": page_num,
            "location": location,
            "item_type": item_type
        })

    def _parse_markdown_to_items(self, md_content: str) -> List[Dict[str, Any]]:
        """Fallback markdown lines parser when items list is missing."""
        items = []
        lines = md_content.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("#"):
                items.append({
                    "type": "heading",
                    "value": line_str.lstrip("#").strip(),
                    "md": line_str
                })
            else:
                items.append({
                    "type": "text",
                    "value": line_str,
                    "md": line_str
                })
        return items
