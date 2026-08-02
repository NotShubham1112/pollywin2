---
name: output-reporting
description: Turns a list of Findings from compliance-engine into a human-readable, actionable report (Markdown, JSON, or CLI summary), grouped by severity with evidence and suggested fixes. Use as the last step in the pipeline.
---

# Output Reporting

## Purpose
Present `Finding`s to the participant (or organizer) in a form that is easy
to act on: grouped by severity, with the specific evidence and a concrete
suggested fix for every violation or warning, and a clear separate bucket
for anything that needs a human's judgment.

## Input
- `List[Finding]` (from `compliance-engine`)
- `format`: `"markdown"`, `"json"`, or `"cli"`

## Output
- A `Report` object, renderable in the requested format, plus an overall
  `status` summary (e.g. "3 blocking issues — not ready to submit").

## Pseudo-code / Logic

```python
class Report:
    def __init__(self, event_name, findings, ruleset_version=None):
        self.event_name = event_name
        self.ruleset_version = ruleset_version
        self.findings = findings
        self.summary = self._build_summary()

    def _build_summary(self):
        counts = {"blocking": 0, "major": 0, "minor": 0, "advisory": 0}
        for f in self.findings:
            if f.status in ("violation", "warning"):
                counts[f.rule.severity] = counts.get(f.rule.severity, 0) + 1
        ready = counts["blocking"] == 0 and counts["major"] == 0
        return {"counts": counts, "ready_to_submit": ready}

class OutputReportingModule:
    SEVERITY_ORDER = ["blocking", "major", "minor", "advisory"]

    def build_report(self, findings, event_name, ruleset_version=None):
        return Report(event_name, findings, ruleset_version)

    def render(self, report, format="markdown"):
        if format == "json":
            return self._render_json(report)
        if format == "cli":
            return self._render_cli(report)
        return self._render_markdown(report)

    def _render_markdown(self, report):
        lines = [f"# Compliance Report — {report.event_name}"]
        if report.ruleset_version:
            lines.append(f"_Checked against rules version: {report.ruleset_version}_")
        lines.append(self._status_line(report))

        violations = [f for f in report.findings if f.status in ("violation", "warning")]
        manual = [f for f in report.findings if f.status == "manual_review"]
        passed = [f for f in report.findings if f.status == "pass"]

        for severity in self.SEVERITY_ORDER:
            group = [f for f in violations if f.rule.severity == severity]
            if not group:
                continue
            lines.append(f"\n## {severity.capitalize()} issues")
            for f in group:
                lines.append(f"- **[{f.rule.id}] {f.rule.description}**")
                lines.append(f"  - Evidence: {f.evidence}")
                if f.suggestion:
                    lines.append(f"  - Suggested fix: {f.suggestion}")

        if manual:
            lines.append("\n## Needs manual review")
            for f in manual:
                lines.append(f"- [{f.rule.id}] {f.rule.description} — {f.evidence}")

        lines.append(f"\n## Passed ({len(passed)} rules)")
        lines.append(", ".join(f.rule.id for f in passed))

        return "\n".join(lines)

    def _render_json(self, report):
        return {
            "event_name": report.event_name,
            "ruleset_version": report.ruleset_version,
            "summary": report.summary,
            "findings": [
                {
                    "rule_id": f.rule.id,
                    "category": f.rule.category,
                    "severity": f.rule.severity,
                    "status": f.status,
                    "evidence": f.evidence,
                    "suggestion": f.suggestion,
                }
                for f in report.findings
            ],
        }

    def _render_cli(self, report):
        return self._status_line(report)

    def _status_line(self, report):
        c = report.summary["counts"]
        if report.summary["ready_to_submit"]:
            return "✅ No blocking or major issues found."
        return (f"⚠️ {c['blocking']} blocking, {c['major']} major, "
                f"{c['minor']} minor, {c['advisory']} advisory issue(s) found.")
```

## Notes
- `manual_review` findings are always shown, never dropped — hiding
  uncertainty is worse than surfacing it.
- The report should let a participant/organizer annotate a finding as
  "acknowledged / intentional, see disclosure" without needing a code
  change, so the tool supports judgment calls rather than overriding them.
- `ready_to_submit` intentionally ignores `minor`/`advisory` severities —
  those inform but don't block.
