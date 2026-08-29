# Knowledge base

Source documents for the RAG assistant. Everything here is **original,
synthetic demo content** written for this project — no proprietary or
copyrighted healthcare material.

```
knowledge/
├── lab_tests/   # what each panel measures, terminology, sample handling
├── sop/         # standard operating procedures for lab workflow
└── faq/         # patient-facing frequently asked questions
```

Supported formats: `.md`, `.txt`, `.pdf`.

Ingest everything with:

```bash
docker compose exec ai-service python -m app.rag.ingest
```

> Informational only. Nothing here is medical advice or a diagnosis.
