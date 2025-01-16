import os
import subprocess

def run_pipeline_for_all_chembl_folders(base_dir, config_path, nproc, ngpus):
    for folder in os.listdir(base_dir):
        if folder.startswith('CHEMBL'):
            if 'novel_aiz_routes.hdf5' in os.listdir(os.path.join(base_dir, folder)):
                print(f"Skipping {folder} as it has already been processed.")
                continue
            targets_path = os.path.join(base_dir, folder, 'actives.smi')
            output_dir = os.path.join(base_dir, folder)
            command = [
                'python', 'src/pipeline.py',
                '--targets', targets_path,
                '--config', config_path,
                '--output_dir', output_dir,
                '--nproc', str(nproc),
                '--ngpus', str(ngpus)
            ]
            print(f"Running command: {' '.join(command)}")
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while processing {folder}: {e.stderr}")

if __name__ == "__main__":
    base_dir = 'paper/01_overlook_popular/rxnutils/kinase'
    config_path = 'paper/01_overlook_popular/rxnutils/rm_config.yml'
    nproc = 7
    ngpus = 1

    run_pipeline_for_all_chembl_folders(base_dir, config_path, nproc, ngpus)