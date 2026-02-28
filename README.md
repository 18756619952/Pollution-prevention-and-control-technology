# Pollution-prevention-and-control-Guidelines
# Supplementary Information for:  
**"Integrating Fine-Tuning and Retrieval-Augmented Generation to Address the Application Challenges of Pollution Control Guidelines"**

This repository contains supplementary materials associated with the manuscript submitted to *Journal of Environmental Management.

---

## 🔧 Repository Structure

- `Pollution Control Guidelines/`: Source regulatory documents used to construct the knowledge base for model training.
- `fine-tune datasets--alpaca--.json`: Structured fine-tuning dataset adapted to the Alpaca format.
- `Evaluation Dataset.json`: Benchmark data used for performance evaluation.
- `evaluate.py`: Python script for computing classical generation metrics and semantic similarity.
- `text segmentation workflow.yml`: Workflow for rule-based and AI-assisted text segmentation（Please import in Dify）.

---
## 🧠 Model

The model data fine tuned based on Llama-factory has been uploaded to HuggingFace：https://huggingface.co/zzzzzzzzzzH/EPCG/tree/main

If you need to run it, please copy the data folder to the save file in Llama-Factory and merge it with the Qwen3-8B model for export.

---

```bash

