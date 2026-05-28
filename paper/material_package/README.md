# Paper Material Package

This package is meant to be the source-of-truth context for paper writing with ChatGPT or other writing assistants. It separates verified implementation facts from hypotheses, experiment plans, and writing prompts.

Recommended use:

1. Start with `00_full_research_lineage.md` when reconstructing the full research story, including failed attempts and conversation-derived reasoning.
2. Use `01_chatgpt_context.md` when asking ChatGPT to draft or polish text.
3. Use `02_method_and_code_inventory.md` when the writing needs exact model details.
4. Use `03_experiment_plan.md` when planning runs, ablations, or result analysis.
5. Use `04_visualization_plan.md` when explaining relation-pattern/head-parameter visualization.
6. Use `05_writing_prompts.md` as reusable prompts.
7. Use `06_run_command_templates.md` when launching the current main experiments.
8. Use `07_ai_writing_workflow.md` to coordinate Codex, ChatGPT, and human revision.

Important boundary:

The current strongest claim should not be "hypernetworks always improve performance." The defensible claim is narrower: relation-conditioned dynamic decision heads can improve convergence or performance on maps where the local interaction rule changes with the observed coordination relation; on easier maps, fixed heads may already be sufficient.
