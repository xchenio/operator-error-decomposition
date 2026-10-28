# Project working agreement

- Commit each coherent code/documentation change after relevant verification; do not commit unrelated user edits.
- Preserve historical data and results. New audits go into new versioned result directories.
- Record commands, evidence, limitations and next steps in docs/RUN_LOG.md.
- Activate the `pytorch` conda environment (`environment.yml`) before running anything and
  invoke plain `python`; do not write absolute interpreter paths into new files, docs or
  scripts. Historical logs that already contain them are records of what ran and stay as they
  are.
- Long training must checkpoint model, optimizer, stage/step and random states. Persist output frequently.
- Never describe legacy truncated captured-energy differences as certified error decompositions.
- Unknown historical code fingerprints remain unknown; do not invent provenance.
