---
doc_id: demo_gocritic_thread
source_type: confluence
title: gocritic paramTypeCombine rule — engineering discussion
date: 2025-09-18
participants: [sam@acmeco.io, priya@acmeco.io, dev-platform@acmeco.io]
skill: decide
---
Long-running engineering discussion about whether to keep the gocritic
linter rule `paramTypeCombine` enabled in CI. sam@acmeco.io argues the
rule produces excessive noise on test files where context parameters
intentionally vary. priya@acmeco.io defends the rule as catching real
duplication in production code. Compromise proposals so far: scope the
rule to non-test packages, or downgrade it from error to warning. No
consensus has been reached as of 2025-09-18.
