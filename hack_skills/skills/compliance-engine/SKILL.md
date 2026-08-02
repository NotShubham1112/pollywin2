---
name: compliance-engine
description: Runs each Rule from a RuleSet against a ProjectBundle using a pluggable CheckerRegistry, producing a Finding (pass/violation/warning/manual_review) per rule with evidence and a suggested fix. Use after rule-ingestion and project-intake.
---

# Compliance Engine

## Purpose
Evaluate a project against the event's rules. For each `Rule`, dispatch to
the matching `Checker` function and produce a `Finding` describing the
result, the evidence for it, and (when relevant) a corrective suggestion.

## Input
- `RuleSet` (from `rule-ingestion`)
- `ProjectBundle` (from `project-intake`)

## Output
- `List[Finding]` — one per rule, each tagged `pass`, `violation`,
  `warning`, or `manual_review`

## Pseudo-code / Logic

```python
class Finding:
    def __init__(self, rule, status, evidence, suggestion=None):
        self.rule = rule
        self.status = status          # "pass" | "violation" | "warning" | "manual_review"
        self.evidence = evidence      # what triggered it: file, line, text excerpt, timestamp
        self.suggestion = suggestion  # corrective action text

class CheckerRegistry:
    _checkers = {}

    @classmethod
    def register(cls, check_type):
        def wrapper(fn):
            cls._checkers[check_type] = fn
            return fn
        return wrapper

    @classmethod
    def get(cls, check_type):
        return cls._checkers.get(check_type, cls.manual_review_fallback)

    @staticmethod
    def manual_review_fallback(rule, bundle):
        return Finding(rule, "manual_review",
                        evidence="No automatic checker registered for this rule type.",
                        suggestion="Please verify manually against the rule text.")

class ComplianceEngine:
    def evaluate(self, rule_set, bundle):
        findings = []
        for rule in rule_set.rules:
            checker = CheckerRegistry.get(rule.check_type)
            findings.append(checker(rule, bundle))
        return findings


# --- Example checker implementations -----------------------------------

@CheckerRegistry.register("tech_stack_restriction")
def check_tech_stack(rule, bundle):
    allowed = set(rule.params.get("allowed_languages", []))
    used = set(bundle.metadata.get("languages_detected", []))
    disallowed = used - allowed if allowed else set()
    if disallowed:
        return Finding(rule, "violation",
                        evidence=f"Disallowed tech used: {disallowed}",
                        suggestion=f"Remove/replace: {', '.join(disallowed)}, "
                                   f"or confirm an exemption with organizers.")
    return Finding(rule, "pass", evidence="Tech stack within allowed list.")


@CheckerRegistry.register("code_freeze_timestamp")
def check_deadline(rule, bundle):
    deadline = rule.params["deadline"]
    last_commit = max(c["timestamp"] for c in bundle.commit_log)
    if last_commit > deadline:
        return Finding(rule, "violation",
                        evidence=f"Last commit at {last_commit}, deadline was {deadline}",
                        suggestion="Revert/remove late commits, or request an exception "
                                   "from organizers if permitted.")
    return Finding(rule, "pass", evidence="All commits before deadline.")


@CheckerRegistry.register("originality_check")
def check_originality(rule, bundle):
    disclosed = bundle.metadata.get("preexisting_code_disclosed", False)
    similarity_score = SimilarityAnalyzer.compare_to_known_repos(bundle)
    threshold = rule.params.get("threshold", 0.8)
    if similarity_score > threshold and not disclosed:
        return Finding(rule, "violation",
                        evidence=f"High similarity ({similarity_score:.0%}) to a prior "
                                 f"public repo, undisclosed.",
                        suggestion="Disclose reused code in the README (what was reused "
                                   "and what was built during the event), or remove it.")
    if similarity_score > threshold and disclosed:
        return Finding(rule, "pass",
                        evidence=f"Similarity {similarity_score:.0%}, but disclosed in README.")
    return Finding(rule, "pass", evidence=f"Similarity {similarity_score:.0%}, below threshold.")


@CheckerRegistry.register("team_size_limit")
def check_team_size(rule, bundle):
    max_size = rule.params.get("max_size")
    team_size = bundle.metadata.get("team_size")
    if max_size and team_size and team_size > max_size:
        return Finding(rule, "violation",
                        evidence=f"Team size {team_size} exceeds limit of {max_size}",
                        suggestion="Confirm roster with organizers, or remove team members "
                                   "from the submission.")
    return Finding(rule, "pass", evidence="Team size within limit.")


@CheckerRegistry.register("required_submission_fields")
def check_submission_fields(rule, bundle):
    required = rule.params.get("required_fields", [])
    missing = [f for f in required if not bundle.metadata.get(f)]
    if missing:
        return Finding(rule, "violation",
                        evidence=f"Missing required fields: {missing}",
                        suggestion=f"Add {', '.join(missing)} to the submission before deadline.")
    return Finding(rule, "pass", evidence="All required fields present.")
```

## Notes
- New rule types are added by registering a new `@CheckerRegistry.register`
  function — the engine itself never needs to change.
- `manual_review` is the deliberate default for anything unimplemented or
  low-confidence; it should never silently collapse to `pass`.
- Checkers that rely on heuristics (originality, "meaningfully uses sponsor
  API") should attach a confidence indicator in `evidence` so downstream
  reporting can weight it appropriately.
