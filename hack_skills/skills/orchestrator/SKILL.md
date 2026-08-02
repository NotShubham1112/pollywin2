---
name: orchestrator
description: Wires rule-ingestion, project-intake, compliance-engine, and output-reporting into a single end-to-end run. Use this as the entry point for checking a hackathon project against event rules.
---

# Orchestrator

## Purpose
Provide one entry point that runs the full pipeline: parse rules, collect
the project, evaluate compliance, and produce a report. Also supports a
fast "re-check" path once rules have already been parsed once.

## Input
- Rules source + format (`markdown` | `yaml` | `json`)
- Project source type (`repo_url` | `zip`) + reference
- Optional interactive metadata answers
- Desired report format (`markdown` | `json` | `cli`)

## Output
- A rendered `Report` (see `output-reporting`), plus the intermediate
  `RuleSet` and `ProjectBundle` (so a re-check doesn't need to re-parse
  rules from scratch)

## Pseudo-code / Logic

```python
class HackathonComplianceChecker:
    def __init__(self):
        self.rule_ingestion = RuleIngestionModule()
        self.project_intake = ProjectIntakeModule()
        self.compliance_engine = ComplianceEngine()
        self.output_reporting = OutputReportingModule()
        self._ruleset_cache = None   # enables fast re-checks

    def run(self, rules_source, rules_format,
             project_source_type, project_source_ref,
             interactive_answers=None, report_format="markdown"):

        rule_set = self._get_or_ingest_ruleset(rules_source, rules_format)

        bundle = self.project_intake.collect(
            project_source_type, project_source_ref, interactive_answers
        )

        findings = self.compliance_engine.evaluate(rule_set, bundle)

        report = self.output_reporting.build_report(
            findings, event_name=rule_set.event_name, ruleset_version=rule_set.version
        )
        return self.output_reporting.render(report, format=report_format)

    def recheck(self, project_source_type, project_source_ref,
                interactive_answers=None, report_format="markdown"):
        # Fast path: reuse the already-parsed RuleSet after a participant
        # fixes issues and wants to verify again.
        if not self._ruleset_cache:
            raise RuntimeError("No cached RuleSet — call run() first.")
        return self.run(
            rules_source=None, rules_format=None,
            project_source_type=project_source_type,
            project_source_ref=project_source_ref,
            interactive_answers=interactive_answers,
            report_format=report_format,
        )

    def _get_or_ingest_ruleset(self, rules_source, rules_format):
        if rules_source is None and self._ruleset_cache:
            return self._ruleset_cache
        rule_set = self.rule_ingestion.ingest(rules_source, format=rules_format)
        self._ruleset_cache = rule_set
        return rule_set


# --- Example usage --------------------------------------------------------

checker = HackathonComplianceChecker()

report_md = checker.run(
    rules_source="https://myevent.devpost.com/rules",
    rules_format="markdown",
    project_source_type="repo_url",
    project_source_ref="https://github.com/team/project",
    interactive_answers={
        "team_size": 4,
        "category": "Best Use of AI",
        "tools_declared": ["OpenAI API", "React"],
        "preexisting_code_disclosed": True,
    },
    report_format="markdown",
)
print(report_md)

# Later, after fixes:
report_md_v2 = checker.recheck(
    project_source_type="repo_url",
    project_source_ref="https://github.com/team/project",
)
```

## Notes
- Caching the `RuleSet` after the first parse is what makes iterative
  self-correction cheap — participants can re-run against the project
  repeatedly without re-parsing the rulebook each time.
- This is the natural place to add CLI/web-UI wrappers, since it exposes a
  single `run()`/`recheck()` surface that hides the four underlying skills.
