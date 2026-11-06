import json
import ast

with open('/home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.py') as f:
    full_py = f.read()

# Verify full script parses
ast.parse(full_py)

lines = full_py.splitlines(keepends=True)

# Find line indices for dividers
idx_sec1 = 0
idx_sec2 = [i for i, l in enumerate(lines) if '# 2. Path Discovery' in l][0] - 1
idx_sec3 = [i for i, l in enumerate(lines) if '# 3. Unsupervised Prompt Recovery' in l][0] - 1
idx_sec4 = [i for i, l in enumerate(lines) if '# 4. LoRA24 In-Memory' in l][0] - 1
idx_sec5 = [i for i, l in enumerate(lines) if '# 5. Main Standalone Execution' in l][0] - 1

cell1_lines = lines[:idx_sec2]
cell2_lines = lines[idx_sec2:idx_sec3]
cell3_lines = lines[idx_sec3:idx_sec4]
cell4_lines = lines[idx_sec4:idx_sec5]
cell5_lines = lines[idx_sec5:]

total_lines = len(cell1_lines) + len(cell2_lines) + len(cell3_lines) + len(cell4_lines) + len(cell5_lines)
assert total_lines == len(lines), f"Line mismatch: {total_lines} != {len(lines)}"

md_source = [
    "# Name That LLM: EXP47 Qwen3-8B Centred Twin Head Attribution Kernel\n",
    "\n",
    "**Experiment**: `EXP47_qwen3_8b_centered_twin`\n",
    "**Lineage**: The Usual Suspects / Dhruv Tibarewal / Sovereign Gold Apex (0.672 LB)\n",
    "**Baseline**: EXP46A (Public LB: 0.606)\n",
    "**Addition**: Unsupervised Content-Word Prompt Recovery + Centred Twin Head (`twin_weight=0.35`)\n",
    "**Hardware**: GPU T4 x2 (or P100 / A100)\n",
    "\n",
    "### Strict Pre-Flight Requirements:\n",
    "1. **Competition Data**: `Name That LLM`\n",
    "2. **Base Model**: `qwen-lm/qwen-3` (Variation: `8b-base`, Framework: `Transformers`)\n",
    "3. **Adapters Dataset**: `dhruvtibarewal/nlm-l24-adapters-v5`\n"
]

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': md_source
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': cell1_lines
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': cell2_lines
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': cell3_lines
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': cell4_lines
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': cell5_lines
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbformat': 4,
            'nbformat_minor': 4
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('/home/rj2826/name-the-llm/notebooks/47_qwen_lora_centered_twin.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print('Conversion complete! Verified line count:', total_lines)
