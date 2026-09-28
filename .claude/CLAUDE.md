# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Strategy (NON-NEGOTIABLE)

**Default to using the Task tool with subagents** to minimize context bloat in the main conversation:

- Use `subagent_type=Explore` for codebase exploration, file searches, and understanding code structure
- Use `subagent_type=Plan` for designing implementation approaches
- Use `subagent_type=Bash` for running commands, tests, or git operations
- Use `subagent_type=general-purpose` for complex multi-step research or tasks

**Always parallelize independent work** - launch multiple Task agents in a single message when tasks don't depend on each other.

To quickly navigate through the codebase, see `docs/explain_var_and_orvarx.md` (code walkthrough) and `docs/dml_implementation.md`.
