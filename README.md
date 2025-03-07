# Template-Free & Template-based integration for reaction space exploration 🧪

## Overview
This project extends [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) to effectively explore reaction space through the integration of template-free approaches with template-based tools. For more details on our research, please refer to [paper link].

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
```

2. Create and set up the [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) environment as described in their documentation.

```bash
cd aizynthfinder
conda env create -f env-dev.yml
poetry install -all-extras
cd ..
conda env update -f retromix.yml
```

3. create an .env file to add the path to your retromix dir, and POSTERA api key. This is essential to run the pipeline. 

In the .env file add:

```bash
POSTERA_API_KEY="{postera_api_key}"
RETROMIX_PATH="{retromix_path}"
```

## Pipeline Overview

Our pipeline consists of 3 key phases:

1. Identification of Popular, Overlooked and Novel templates
2. Integration of these templates in AiZynthFinder route searches
3. Scoring of post-optimization search performance

### Phase 1: Template Identification

We provide files to complete this process with a template-based (AiZynthFinder) and template-free (AiZynthFinder or Postera) methods. 

For example to run the AiZynthFinder only method:

Run the template identification script:
```bash
python src/find_templates_aiz.py --rm_config {config_file} --targets {target_file_path} --output_path {output_path} --scoring_type {scoring_type}
```

The default route scoring type is state score as provided by AiZynthFinder. This will rank routes based on their state score, then extract relevant templates. 

### Phase 2: Template Integration

The next step is to search for routes to target molecules while optimsing specfic templates. 

Execute the route search:
```bash
python src/route_search.py --config {rm_config_path} --output_dir {output_dir} --type {optimisation_type} --templates {template_file} --optimise {bool} --smiles {target_smiles}
```

### Phase 3: Performance Scoring

Run the scoring analysis:

The final step is to calculate the terms of the POS score (a, b , y). 

```bash
python src/optimisation/scoring.py --pre {pre_file - aiz} --post {post_file - aiz} --config {rm_config_path} --output {output_file_path}
```

There is functionality to create your own scoring function. Currently, cost, state score, frequency and CoPriNet prediction model (will perform better with gpu access) to rank templates. 

## Run the pipeline
Alternatively, the pipeline.py file allows you to run the whole pipeline in one. 

```bash
python src/pipeline.py --targets {target_mol_path} --config {retromix_config} --output_dir {output_dir}
```

## Dependencies
- AiZynthFinder
- rdcanon
- rdchiral

## Citation
If you use this work, please cite our paper: [paper citation]

