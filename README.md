# Template-Free & Template-based integration for reaction space exploration 🧪

🚧 **This repository is STILL UNDER DEVELOPMENT** 🚧

## Overview
This project extends [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) to effectively explore reaction space through the integration of template-free approaches with template-based tools. For more details on our research, please refer to [paper link].

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
```

2. Create and set up the [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) environment as described in their documentation.

3. Install additional required packages:
```bash
pip install rdcanon rdchiral
```

## Pipeline Overview

Our pipeline consists of 3 key phases:

1. Popular, Overlooked and Novel template identification
2. Integration of these templates in route searches
3. Scoring of post-optimization route search performance

### Phase 1: Template Identification
Navigate to `notebooks/find_templates.ipynb` for detailed walkthrough.

Run the template identification script:
```bash
python src/find_templates.py --rm_config {config_file}
```

### Phase 2: Template Integration
Reference `notebooks/integration.ipynb` for implementation details.

Execute the route search:
```bash
python src/route_search.py {arguments}
```

### Phase 3: Performance Scoring
See `notebooks/pos_scoring.ipynb` for analysis methodology.

Run the scoring analysis:
```bash
python src/analysis/scoring.py --pre {pre_file} --post {post_file} --config {fm_config_file}
```

## Documentation
- Detailed documentation can be found in the respective notebooks:
  - Template Finding: `notebooks/find_templates.ipynb`
  - Integration: `notebooks/integration.ipynb`
  - Post-optimization Scoring: `notebooks/pos_scoring.ipynb`

## Dependencies
- AiZynthFinder
- rdcanon
- rdchiral

## Citation
If you use this work, please cite our paper: [paper citation]

## License
TODO: [Add license information]

## Contact
TODO: [Add contact information]
