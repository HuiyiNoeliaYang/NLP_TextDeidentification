#!/usr/bin/env python3
"""
Show the dataset structure at different stages.
"""

import datasets

def show_raw_dataset():
    """Show raw dataset from HuggingFace."""
    print('=' * 80)
    print('1. RAW DATASET (from HuggingFace, before any processing):')
    print('=' * 80)
    
    dataset = datasets.load_dataset('wiki_bio', split='train[:1]')
    ex = dataset[0]
    
    print(f'\nColumn names: {dataset.column_names}')
    print(f'Keys in example: {list(ex.keys())}')
    print(f'\n❌ Does it have a "name" column? {"name" in ex.keys()}')
    
    print(f'\n"input_text" structure:')
    print(f'  Type: {type(ex["input_text"])}')
    print(f'  Keys: {list(ex["input_text"].keys())}')
    print(f'  Table column_header: {ex["input_text"]["table"]["column_header"]}')
    print(f'  Table content: {ex["input_text"]["table"]["content"]}')
    
    print(f'\n"target_text" (first 200 chars):')
    print(ex['target_text'][:200] + '...')
    
    return ex

def show_processed_dataset():
    """Show dataset after create_document_and_profile."""
    print('\n\n' + '=' * 80)
    print('2. PROCESSED DATASET (after create_document_and_profile):')
    print('=' * 80)
    
    # Simplified version of create_document_and_profile (standalone, no imports)
    def find_row_from_key(table_rows, key):
        """Finds row in wikipedia infobox by a key"""
        matching_rows = [r for r in table_rows if r[0] == key]
        if len(matching_rows) == 0:
            raise AssertionError(f"name not found in table_rows: {table_rows}")
        return matching_rows[0]
    
    def name_from_table_rows(table_rows):
        """gets person's name from rows of a Wikipedia infobox"""
        try:
            row = find_row_from_key(table_rows, key='name')
        except AssertionError:
            # a few articles have this key instead of a 'name' key
            row = find_row_from_key(table_rows, key='article_title')
        name = row[1]
        # capitalize name and return
        return ' '.join((word.capitalize() for word in name.split()))
    
    def create_document_and_profile_simple(ex):
        fixed_target_text = ex['target_text'].replace('-rrb-', ')').replace('-lrb-', '(')
        table_info = ex['input_text']['table'] 
        table_column_header = list(table_info['column_header'])
        table_content = list(table_info['content'])
        
        profile_keys = [str(s).strip().replace('|', '') for s in table_column_header]
        profile_values = [str(s).strip().replace('|', '') for s in table_content]
        table_rows = list(zip(profile_keys, profile_values))
        table_text = '\n'.join([' || '.join(row) for row in table_rows])
        
        return {
            'name': name_from_table_rows(table_rows),  # ✅ NOW INCLUDED!
            'document': fixed_target_text,
            'profile': table_text,
            'profile_keys': '||'.join(profile_keys),
            'profile_values': '||'.join(profile_values),
            'text_key': fixed_target_text + ' ' + table_text,
        }
    
    dataset = datasets.load_dataset('wiki_bio', split='train[:1]')
    processed = dataset.map(create_document_and_profile_simple, num_proc=1)
    ex = processed[0]
    
    print(f'\nColumn names: {processed.column_names}')
    print(f'Keys in example: {list(ex.keys())}')
    print(f'\n✅ Does it have a "name" column? {"name" in ex.keys()}')
    
    print(f'\n"name": {ex["name"]}')
    print(f'\n"document" (first 200 chars): {ex["document"][:200]}...')
    print(f'\n"profile":')
    print(ex["profile"])
    print(f'\n"profile_keys": {ex["profile_keys"]}')
    print(f'\n"profile_values": {ex["profile_values"]}')
    
    return ex

if __name__ == "__main__":
    try:
        raw_ex = show_raw_dataset()
        processed_ex = show_processed_dataset()
        
        print('\n\n' + '=' * 80)
        print('SUMMARY:')
        print('=' * 80)
        print('Raw dataset: NO "name" column ❌')
        print('  - Name is embedded in input_text["table"]["column_header"] and ["content"]')
        print('  - You need to extract it from profile_keys/profile_values')
        print()
        print('Processed dataset: HAS "name" column ✅')
        print('  - Name is extracted by create_document_and_profile()')
        print('  - Uses name_from_table_rows() which tries "name" then "article_title"')
        print('  - Can access directly: dataset["name"]')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

