#!/usr/bin/env python3
"""
Script to inspect the WikiBio dataset structure and examples.
"""

import datasets

def create_document_and_profile_simple(ex):
    """Simplified version of create_document_and_profile for inspection."""
    fixed_target_text = ex['target_text'].replace('-rrb-', ')').replace('-lrb-', '(')
    table_info = ex['input_text']['table'] 
    table_column_header = list(table_info['column_header'])
    table_content = list(table_info['content'])
    
    profile_keys = [str(s).strip().replace('|', '') for s in table_column_header]
    profile_values = [str(s).strip().replace('|', '') for s in table_content]
    table_rows = list(zip(profile_keys, profile_values))
    table_text = '\n'.join([' || '.join(row) for row in table_rows])
    
    return {
        'document': fixed_target_text,
        'profile': table_text,
        'profile_keys': '||'.join(profile_keys),
        'profile_values': '||'.join(profile_values),
        'text_key': fixed_target_text + ' ' + table_text,
    }

def inspect_raw_wikibio():
    """Inspect the raw WikiBio dataset from HuggingFace."""
    print("=" * 80)
    print("Loading raw WikiBio dataset from HuggingFace...")
    print("=" * 80)
    
    # Load a small sample
    try:
        dataset = datasets.load_dataset('wiki_bio', split='train[:5]', version='1.2.0')
    except (TypeError, RuntimeError) as e:
        if "version" in str(e) or "no longer supported" in str(e):
            try:
                dataset = datasets.load_dataset('wiki_bio', split='train[:5]', trust_remote_code=True)
            except Exception:
                dataset = datasets.load_dataset('wiki_bio', split='train[:5]')
        else:
            raise
    
    print(f"\nDataset type: {type(dataset)}")
    print(f"Number of examples: {len(dataset)}")
    print(f"\nDataset features (columns): {dataset.features}")
    print(f"\nColumn names: {dataset.column_names}")
    
    print("\n" + "=" * 80)
    print("Example 0 (raw):")
    print("=" * 80)
    example = dataset[0]
    print(f"\nKeys: {list(example.keys())}")
    
    print(f"\n'target_text' (first 200 chars):")
    print(example['target_text'][:200] + "...")
    
    print(f"\n'input_text' structure:")
    print(f"  Type: {type(example['input_text'])}")
    if isinstance(example['input_text'], dict):
        print(f"  Keys: {list(example['input_text'].keys())}")
        if 'table' in example['input_text']:
            print(f"  'table' keys: {list(example['input_text']['table'].keys())}")
            print(f"  'table' column_header: {example['input_text']['table']['column_header']}")
            print(f"  'table' content: {example['input_text']['table']['content']}")
    
    print("\n" + "=" * 80)
    print("All examples (raw):")
    print("=" * 80)
    for i in range(len(dataset)):
        ex = dataset[i]
        print(f"\n--- Example {i} ---")
        print(f"Column headers: {ex['input_text']['table']['column_header']}")
        print(f"Content: {ex['input_text']['table']['content']}")
        print(f"Target text (first 100 chars): {ex['target_text'][:100]}...")

def inspect_processed_wikibio():
    """Inspect the processed WikiBio dataset (after create_document_and_profile)."""
    print("\n\n" + "=" * 80)
    print("Loading and processing WikiBio dataset...")
    print("=" * 80)
    
    # Load and process
    try:
        dataset = datasets.load_dataset('wiki_bio', split='train[:5]', version='1.2.0')
    except (TypeError, RuntimeError) as e:
        if "version" in str(e) or "no longer supported" in str(e):
            try:
                dataset = datasets.load_dataset('wiki_bio', split='train[:5]', trust_remote_code=True)
            except Exception:
                dataset = datasets.load_dataset('wiki_bio', split='train[:5]')
        else:
            raise
    processed_dataset = dataset.map(create_document_and_profile_simple, num_proc=1)
    
    print(f"\nProcessed dataset type: {type(processed_dataset)}")
    print(f"Number of examples: {len(processed_dataset)}")
    print(f"\nProcessed dataset features (columns): {processed_dataset.features}")
    print(f"\nColumn names: {processed_dataset.column_names}")
    
    print("\n" + "=" * 80)
    print("Example 0 (processed):")
    print("=" * 80)
    example = processed_dataset[0]
    print(f"\nKeys: {list(example.keys())}")
    
    print(f"\n'document' (first 200 chars):")
    print(example['document'][:200] + "...")
    
    print(f"\n'profile':")
    print(example['profile'])
    
    print(f"\n'profile_keys':")
    print(example['profile_keys'])
    
    print(f"\n'profile_values':")
    print(example['profile_values'])
    
    print(f"\n'text_key' (first 200 chars):")
    print(example['text_key'][:200] + "...")
    
    print("\n" + "=" * 80)
    print("All examples (processed):")
    print("=" * 80)
    for i in range(len(processed_dataset)):
        ex = processed_dataset[i]
        print(f"\n--- Example {i} ---")
        print(f"Profile keys: {ex['profile_keys']}")
        print(f"Profile values: {ex['profile_values']}")
        print(f"Document (first 150 chars): {ex['document'][:150]}...")
        print(f"Profile: {ex['profile']}")

def inspect_with_datamodule():
    """Inspect using the project's DataModule (most realistic)."""
    print("\n\n" + "=" * 80)
    print("Loading WikiBio using WikipediaDataModule...")
    print("=" * 80)
    print("(Skipping - requires transformers and other dependencies)")
    print("Run this in your environment with all dependencies installed.")

if __name__ == "__main__":
    try:
        inspect_raw_wikibio()
    except Exception as e:
        print(f"Error inspecting raw dataset: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        inspect_processed_wikibio()
    except Exception as e:
        print(f"Error inspecting processed dataset: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        inspect_with_datamodule()
    except Exception as e:
        print(f"Error inspecting with DataModule: {e}")
        import traceback
        traceback.print_exc()

