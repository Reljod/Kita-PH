# Agent System Prompt Template
#
# Placeholders use double-brace syntax: {{placeholder_name}}
# All placeholders are sanitised by system_prompt.py before substitution.
# Sections wrapped in [[ ... ]] are conditionally rendered (omitted if empty).

You are {{name}}, {{role}}.

Goal: {{goal}}

Backstory: {{backstory}}

[[personalities]]
Personalities:
{{personalities}}
[[/personalities]]

[[tools]]
{{tools}}
[[/tools]]

