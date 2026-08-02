---
name: rule-ingestion
description: Parses a hackathon's rules (Markdown/PDF/plain-text prose, or pre-structured YAML/JSON) into a normalized, machine-checkable RuleSet. Use this first in the pipeline, before project-intake or compliance-engine.
---

# Rule Ingestion

## Purpose
Convert the hackathon's rules — usually unstructured prose from a rulebook
page or PDF — into a structured, machine-checkable rule set. Also accepts a
pre-structured YAML/JSON rules file directly, for organizers who want to
skip NLP entirely and guarantee accuracy.

## Input
- Raw rules document: Markdown, PDF-extracted text, or plain text (scraped
  from the event website or rulebook)
- OR a structured override file: YAML or JSON, pre-authored by organizers,
  matching the `Rule` schema below

## Output
A `RuleSet`: an object holding the event name and a list of `Rule` items,
each with `id`, `category`, `description`, `check_type`, `severity`, and
`params`.

## Pseudo-code / Logic

```python
class Rule:
    def __init__(self, id, category, description, check_type, severity, params=None):
        self.id = id                      # e.g. "R-01"
        self.category = category          # "tech_stack", "submission", "originality", "team", "scope"
        self.description = description    # human-readable rule text
        self.check_type = check_type      # maps to a Checker in compliance-engine
        self.severity = severity          # "blocking", "major", "minor", "advisory"
        self.params = params or {}        # e.g. {"allowed_languages": ["Python", "JS"]}

class RuleSet:
    def __init__(self, event_name, rules: list[Rule], version=None):
        self.event_name = event_name
        self.rules = rules
        self.version = version            # timestamp/tag, rulebooks can change mid-event

class RuleIngestionModule:
    def ingest(self, source, format="markdown"):
        if format in ("yaml", "json"):
            raw = self.load_structured(source)
            return self.build_ruleset_from_structured(raw)
        else:
            text = self.load_text(source)
            candidate_rules = self.extract_rule_statements(text)
            structured = [self.classify_and_structure(r) for r in candidate_rules]
            return RuleSet(event_name=self.detect_event_name(text), rules=structured)

    def extract_rule_statements(self, text):
        # Split by headings/bullets; keep sentences containing normative
        # language ("must", "may not", "should", "required", "prohibited",
        # "no more than", "prior to").
        ...

    def classify_and_structure(self, statement):
        # NLP/LLM classification into category + check_type + params.
        # Low-confidence extractions fall back to:
        #   check_type = "manual_review", severity = "advisory"
        # rather than guessing.
        ...

    def build_ruleset_from_structured(self, raw):
        rules = [Rule(**r) for r in raw["rules"]]
        return RuleSet(event_name=raw.get("event_name"), rules=rules,
                        version=raw.get("version"))
```

## Notes
- Prefer the structured YAML/JSON path whenever organizers can provide it —
  it removes the single biggest source of error in the whole pipeline.
- Every `Rule` produced from prose should retain the original sentence as
  provenance (not shown above for brevity) so reports can quote back "why"
  a rule exists.
