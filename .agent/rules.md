# BioReef.ai Cognitive Architecture Rules

## 1. Knowledge Base
- Always reference `@.context/upgraded_solution.md` for architectural logic.
- Always reference `@.context/stage_1_preprocessing.md` for data handling.
- Always reference `@.context/progress_report_1.md` for target metrics and HOTA/HD goals.

## 2. Implementation Guardrails
- **No Generic Logic:** Do not use standard CNN object detectors. Every detection must use the 4-stream Context Harvester and MCEAM Fusion.
- **Biological Consistency:** Every Stage 3 output must pass through the Taxonomic Guardrail (Consistency Mask) defined in the solution spec.
- **Evaluation:** Every tracking update must include a HOTA validation script. Every classification update must include a Hierarchical Distance (HD) log.

## 3. Code Style
- Use PyTorch for all model definitions.
- Use Hydra or YAML for configuration management.
- Documentation must follow the "Marine Biologist" persona: accurate, taxonomic, and ecologically grounded.