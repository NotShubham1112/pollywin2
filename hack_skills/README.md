# Hackathon Project Rule-Compliance Checker

A conceptual design, packaged as a set of "skills" (one per component/module).
Each skill lives in `skills/<name>/SKILL.md` and follows the same shape:
Purpose → Input → Output → Pseudo-code. This file covers the parts that span
all of them: the overarching goal, how the skills chain together, and the
open design risks.

## 1. Goal

Automatically evaluate a hackathon submission — code repository,
documentation, and metadata — against a given event's official rules and
guidelines (eligibility, tech-stack restrictions, submission format,
originality/plagiarism policy, category constraints). The system surfaces
potential violations with severity levels, explains *why* each is flagged,
and suggests concrete corrective actions — so participants can self-correct
before the deadline and organizers can triage entries faster.

## 2. Skills in this set

| Skill | Role |
|---|---|
| [`rule-ingestion`](skills/rule-ingestion/SKILL.md) | Turns prose rules (MD/PDF/plain text) or structured YAML/JSON into a machine-checkable `RuleSet` |
| [`project-intake`](skills/project-intake/SKILL.md) | Collects the participant's repo, docs, and metadata into a `ProjectBundle` |
| [`compliance-engine`](skills/compliance-engine/SKILL.md) | Runs each `Rule` against the `ProjectBundle` and produces `Finding`s |
| [`output-reporting`](skills/output-reporting/SKILL.md) | Turns `Finding`s into a human-readable report with corrective actions |
| [`orchestrator`](skills/orchestrator/SKILL.md) | Wires the above together into one end-to-end run |

## 3. Overall Program Flow

```
 1. Organizer or participant supplies:
      - Rules source (rulebook URL/file, or pre-made YAML/JSON)
      - Project source (repo URL / ZIP) + a short interactive Q&A
            │
            ▼
 2. rule-ingestion.ingest(rules_source)        →  RuleSet
            │
 3. project-intake.collect(project_source)     →  ProjectBundle
            │
            ▼
 4. compliance-engine.evaluate(RuleSet, ProjectBundle)
      for each Rule:
          checker = CheckerRegistry.get(rule.check_type)
          finding = checker(rule, bundle)
      → List[Finding]
            │
            ▼
 5. output-reporting.build_report(findings)
      - group by severity (blocking > major > minor > advisory)
      - attach evidence + suggested fix per finding
      - flag "manual_review" items separately
      → Report (Markdown / JSON / CLI summary)
            │
            ▼
 6. Participant fixes flagged items → re-run from step 3 (fast path,
    rules already parsed) until clean, or until deadline.
```

The `orchestrator` skill implements this as a single callable pipeline; see
that skill for the driving pseudo-code.

## 4. Key Challenges & Considerations

- **Rule parsing accuracy (NLP).** Hackathon rulebooks are prose, often
  ambiguous ("teams should primarily use...", "sponsor tech is encouraged
  but not required"). Misclassifying a rule's `check_type` or `severity`
  risks false confidence. Mitigation: low-confidence extractions default to
  `manual_review` rather than silently passing or failing; allow organizers
  to supply a structured YAML/JSON override that bypasses NLP entirely.

- **Depth vs. feasibility of project analysis.** Some checks are shallow
  (deadline timestamp, language list from manifest files); others imply deep
  static/semantic analysis (originality/plagiarism, "was this built during
  the event", "does it actually use the sponsor API meaningfully"). Fully
  automating the deep checks is out of reach for most implementations —
  the design should aim for *decision support*, not a final verdict.

- **False positives / false negatives.** A false "violation" erodes trust
  and wastes participant time; a false "pass" lets a real violation through
  to judging. Every automated check should carry a confidence level and
  cite its evidence, and blocking-severity findings should be the ones held
  to the highest precision bar (favor `manual_review` over a wrong
  `violation` when uncertain).

- **Originality/plagiarism detection limits.** Comparing against "known
  public repos" is itself an open research problem (which corpus? how much
  similarity is expected boilerplate vs. copied work?). This is best framed
  as a similarity *signal* that triggers manual review, not an automatic
  disqualification.

- **Extensibility.** New hackathons bring new rule types. The
  `CheckerRegistry` pattern (register a function per `check_type`) keeps new
  checks pluggable without touching the engine; unregistered `check_type`s
  fall back to `manual_review` instead of crashing or being silently
  skipped.

- **Data collection friction / privacy.** Cloning a private repo, reading
  commit history, and asking participants to self-disclose (e.g. pre-existing
  code, team composition) needs clear consent and minimal-necessary access
  — especially if the tool is run by organizers on participants' behalf
  rather than by participants themselves.

- **User experience.** Findings must read like a helpful pre-submission
  checklist, not a rejection notice — clear evidence, a concrete suggested
  fix, and a way to dismiss/annotate a finding as "intentional, see
  disclosure" rather than forcing a code change.

- **Versioning of rules.** Rulebooks change during an event (clarifications,
  FAQ updates). The `RuleSet` should carry a version/timestamp so a report
  can state which version of the rules it was checked against.
