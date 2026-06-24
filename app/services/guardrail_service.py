from typing import Any, Dict

DEFAULT_GUARDRAILS: Dict[str, Any] = {
    "max_delegation_depth": 5,
    "max_websearch_depth": 10,
}

def resolve_guardrails(agent_config: Dict[str, Any], org_config: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(DEFAULT_GUARDRAILS)
    for key in result:
        if key in agent_config:
            result[key] = agent_config[key]
        elif key in org_config:
            result[key] = org_config[key]
    return result
