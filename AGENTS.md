- Mamba env: 3dml
## Corrections & preferences
- **Prompts:** Always use view-dependent prompts (`prompt.view_dependent: true`, per-view qualifiers) in vdsds training runs. _Why:_ plain duplicated prompts with per-view-deformable geometry produce Janus (a second head forming on the back); per-view canonical conditioning prevents it. (2026-11-20)
- **Runs:** Don't start training runs (or queue follow-up runs) without an explicit user request; diagnose, report findings and propose, then launch only on confirmation. _Why:_ launched houndhippo_card and pledged an 8-view run unprompted while answering a question. (2026-11-20)
