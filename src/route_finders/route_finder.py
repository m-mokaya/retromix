import os
import sys

import pandas as pd
import multiprocessing as mp


class RouteFinder:
    """
    A general class for finding retrosynthetic routes using different tools.
    """

    def __init__(self, smiles: list, nproc: int, output_dir: str):
        """
        Initializes the RouteFinder class.

        :param smiles: A list of SMILES strings representing the target molecules.
        :param nproc: The number of processes to use for parallel processing.
        :param output: The directory to save the results to.
        """
        self.smiles = smiles
        self.nproc = nproc
        self.output_dir = output_dir

    def split_smiles(self) -> list[list[str]]:
        """
        Splits the list of SMILES strings into chunks for parallel processing.

        :return: A list of chunks, where each chunk is a list of SMILES strings.
        """
        if len(self.smiles) == 0:
            print('No SMILES strings to process')
            return []
        elif len(self.smiles) == 1:
            return [self.smiles]
        else:
            chunk_size = len(self.smiles) // self.nproc
            remainder = len(self.smiles) % self.nproc
            chunks = [self.smiles[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(self.nproc)]
            return chunks

    def save_results(self, results_df, filename):
        """
        Saves the results to a file.

        :param results_df: A pandas DataFrame containing the results.
        :param filename: The name of the file to save the results to.
        """
        results_df.to_hdf(os.path.join(self.output_dir, filename), key='table', mode='w')
    
    def find_routes(self, filename):
        """
        Finds retrosynthetic routes for the target molecules.

        :param filename: The name of the file to save the results to.
        :return: A pandas DataFrame containing the results.
        """
        raise NotImplementedError("This method must be implemented in the child class.")

    def convert_results_to_aiz_format(self, data):
        """
        Converts the results to the format used by AiZynthFinder.

        :param data: The results data to convert.
        :return: The converted results data.
        """
        raise NotImplementedError("This method must be implemented in the child class.")