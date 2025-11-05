# Lightning AI Setup Checklist

This document provides a step-by-step guide to run the deidentification script on Lightning AI.

## ✅ Pre-flight Checklist

### 1. Upload Project Files
- [ ] Upload the entire project folder to Lightning AI
- [ ] Ensure all files are in the correct directory structure

### 2. Model Checkpoint
- [ ] Download the model from HuggingFace:
  ```bash
  git clone https://huggingface.co/jxm/wikibio_roberta_roberta_idf
  ```
  This will create the `wikibio_roberta_roberta_idf/` folder in your project root with the `model.ckpt` file inside.
- [ ] Verify the checkpoint exists at: `./wikibio_roberta_roberta_idf/model.ckpt`
- [ ] OR update `model_cfg.py` line 40 if using a different path

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** The following packages are already in `requirements.txt`:
- ✅ `textattack==0.3.10`
- ✅ `datasets==2.14.7` (compatible version)
- ✅ `pytorch-lightning==1.9.4`
- ✅ `transformers`, `torch`, etc.

### 4. Install spaCy Model (if not already installed)
```bash
python -m spacy download en_core_web_sm
```

### 5. Run the Deidentification Script
```bash
python scripts/deidentify.py --model_key model_8_ls0.01 --n 1 --k 1
```

## 📝 Expected Behavior

1. **First Run (Slow):**
   - Downloads WikiBio dataset (~2.3GB) - will be faster on Lightning AI
   - Tokenizes entire training set (72,831 examples) - faster with more CPUs
   - Precomputes profile embeddings if they don't exist
   - Estimated time: 30-60 minutes (vs 1.5-2 hours on Mac)

2. **Subsequent Runs (Fast):**
   - Uses cached dataset and embeddings
   - Only processes the documents you specify (`--n 1`)
   - Estimated time: 1-5 minutes

## 🔧 Troubleshooting

### Issue: Model checkpoint not found
**Solution:** Check `model_cfg.py` line 40 and ensure the checkpoint path is correct.

### Issue: Dataset loading error
**Solution:** The code has been updated to handle dataset compatibility. If issues persist, try:
```bash
pip install "datasets<2.15.0"
```

### Issue: Module not found
**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Slow tokenization
**Solution:** This is normal on first run. Lightning AI should be faster than local Mac due to more CPU cores.

## 📂 Output Files

- **Masked documents:** Saved in `adv_csvs_full_8/` directory
- **Profile embeddings:** Saved in `embeddings/profile/{model_key}/` directory
- **Logs:** Check terminal output for progress

## ✅ What's Already Fixed

1. ✅ `sys.path` - Uses dynamic paths (no hardcoded paths)
2. ✅ `os.sched_getaffinity` - Handles macOS/Linux compatibility
3. ✅ CPU/GPU detection - Automatically detects available device
4. ✅ Checkpoint loading - Handles state_dict mismatches with `strict=False`
5. ✅ Dataset loading - Handles newer datasets library versions
6. ✅ `num_cpus` in `utils/embedding.py` - Now uses all available CPUs

## 🚀 Ready to Run!

Once all items are checked, you're ready to run the deidentification script on Lightning AI!

