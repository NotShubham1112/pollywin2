---
name: project-intake
description: Collects a participant's project artifacts (repo/ZIP, README, dependency manifests, commit history) and metadata (via interactive prompts) into a normalized ProjectBundle. Use after rule-ingestion, before compliance-engine.
---

# Project Intake

## Purpose
Collect the participant's project — code, docs, and metadata — into a
normalized form the compliance engine can evaluate.

## Input
- A repo URL (GitHub/GitLab) or an uploaded ZIP archive
- Interactive prompt answers: team size, category entered, tools/sponsor
  tech used, third-party assets used, hours worked, pre-existing code
  disclosure
- Optional: a submission-page URL (e.g. Devpost) to auto-scrape the
  description and demo video link

## Output
A `ProjectBundle`: normalized file tree, README/docs text, dependency
manifests, commit history metadata, and declared metadata (team info,
category, disclosures).

## Pseudo-code / Logic

```python
class ProjectBundle:
    def __init__(self):
        self.files = {}            # path -> content (or lazy loader)
        self.readme_text = ""
        self.dependencies = []     # from package.json, requirements.txt, etc.
        self.commit_log = []       # [{"timestamp": ..., "author": ...}, ...]
        self.metadata = {}         # team_size, category, tools_declared, etc.

class ProjectIntakeModule:
    def collect(self, source_type, source_ref, interactive_answers=None):
        bundle = ProjectBundle()

        if source_type == "repo_url":
            local_path = self.clone_repo(source_ref)
            bundle.files = self.walk_files(local_path)
            bundle.commit_log = self.extract_git_log(local_path)
        elif source_type == "zip":
            bundle.files = self.extract_zip(source_ref)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

        bundle.readme_text = self.find_and_read_readme(bundle.files)
        bundle.dependencies = self.parse_manifests(bundle.files)
        bundle.metadata = interactive_answers or self.prompt_user_for_metadata()
        bundle.metadata["languages_detected"] = self.detect_languages(bundle.files)
        return bundle

    def prompt_user_for_metadata(self):
        # Short interactive Q&A, e.g.:
        #   - team size
        #   - declared category
        #   - sponsor tech / APIs used
        #   - hours worked / start time
        #   - pre-existing code disclosure (yes/no + description)
        ...

    def find_and_read_readme(self, files):
        ...

    def parse_manifests(self, files):
        # package.json, requirements.txt, pyproject.toml, go.mod, Cargo.toml, etc.
        ...

    def detect_languages(self, files):
        # file-extension histogram, or a lightweight language-detection lib
        ...
```

## Notes
- Cloning private repos or reading commit history requires explicit
  participant consent — surface what will be read before pulling it.
- Keep intake idempotent/cheap to re-run: after a participant fixes an
  issue, the fast path is re-collecting the project (rules are already
  parsed) and re-running the compliance engine.
