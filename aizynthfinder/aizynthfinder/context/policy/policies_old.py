""" 
Module containing classes that interfaces neural network policies.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
# from aizynthfinder.context.policy.custom_expansion_policy import CustomExpansionModel

from aizynthfinder.context.collection import ContextCollection
from aizynthfinder.context.policy.expansion_strategies import (
    ExpansionStrategy,
    TemplateBasedExpansionStrategy,
)
from aizynthfinder.context.policy.expansion_strategies import (
    __name__ as expansion_strategy_module,
)
from aizynthfinder.context.policy.filter_strategies import (
    FILTER_STRATEGY_ALIAS,
    FilterStrategy,
    QuickKerasFilter,
)
from aizynthfinder.context.policy.filter_strategies import (
    __name__ as filter_strategy_module,
)
from aizynthfinder.utils.exceptions import PolicyException
from aizynthfinder.utils.loading import load_dynamic_class

if TYPE_CHECKING:
    from aizynthfinder.chem import TreeMolecule
    from aizynthfinder.chem import TemplatedRetroReaction
    from aizynthfinder.chem.reaction import RetroReaction
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.utils.type_utils import Any, List, Sequence, Tuple

from aizynthfinder.chem import TemplatedRetroReaction
from aizynthfinder.chem import TreeMolecule

import copy
import torch
import pandas as pd
import json
import sys

from rdkit import Chem
from rdchiral import main as rdc

class ExpansionPolicy(ContextCollection):
    """
    An expansion policy that uses a neural network to predict the next expansion. Here we modify or extend the templates accessible to the network.
    :param config: the aizynthfinder configuration settings
    :param template_library: the library of templates
    """
    
    _collection_name = "expansion_policy"
    
    def __init__(self, config: Configuration) -> None:
        super().__init__()
        self._config = config
        
    def __call__(
        self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        return self.get_actions(molecules)
    
    def _get_optimised_templates(self, novel: bool) -> list[tuple[str, float]]:
        """
        Get the optimised templates from the optimisation data.
        
        :param novel: if True, get the novel templates, otherwise get the popular or overlooked templates
        """
        if novel:
            if self._config.novel_template_data is not None:
                with open(self._config.novel_template_data, 'r') as f:
                    optimised_templates = json.load(f)
                optimised_templates = [(k, v) for k, v in optimised_templates.items()]
                return optimised_templates
            else:
                raise PolicyException("No optimisation data for novel templates")
        else:
            if self._config.optimisation_data is not None: 
                with open(self._config.optimisation_data, 'r') as f:
                    optimised_templates = json.load(f)
                optimised_templates = [(k, v) for k, v in optimised_templates.items()]
                return optimised_templates
            else:
                raise PolicyException("No optimisation data for popular templates")
        
    def _check_foreign_compatibility(self, foreign_templates: list[tuple], molecules: TreeMolecule) -> list[str]:
        """
        Check if the foreign templates are compatible with the target molecules. 
        
        :param foreign_templates: the foreign templates to check
        :param molecules: the target molecules
        :return: the compatible templates
        """
        compatible_templates = []
        for mol in molecules:
            for rxn, score in foreign_templates:
                reaction = rdc.rdchiralReaction(rxn)
                reactants = rdc.rdchiralReactants(mol.smiles)
                try:
                    outcomes = rdc.rdchiralRun(reaction, reactants)
                    if outcomes != None:
                        compatible_templates.append(rxn)
                except:
                    continue
            return compatible_templates
        
    def _integrate_foreign_templates(self, possible_actions, priors, foreign_templates):
        """
        Integrate foreign templates into the possible actions.
        
        :param possible_actions: the possible actions
        :param priors: the priors of the possible actions
        :param foreign_templates: the foreign templates to integrate
        :return: the updated possible actions and priors
        """
        max_prior = max(priors) if priors else 0.0
        min_prior = min(priors) if priors else 0.0
        prior_range = max_prior - min_prior
        
        for template, score in foreign_templates:
            new_action = possible_actions[0].copy()     # copy existing actions as a base
            new_action.metadata['template'] = template
            new_action.metadata['opt_score'] = score
            new_action.metadata['classification'] = 'novel'
            new_action.metadata['policy_name'] = 'external'
            
            # estimate prior for foreign template based on opt score, prior range and min prior
            estimated_prior = min_prior + (score * prior_range)
            
            # calculate opt factor, and new prior
            optimisation_factor = score * prior_range
            new_prior = estimated_prior + optimisation_factor
            new_action.metadata['policy_probability'] = new_prior
            
            possible_actions.append(new_action)
            priors.append(new_prior)
            
        # normalise and reorder the priors
        norm_priors = [float(i)/sum(priors) for i in priors]
        possible_actions = [x for _, x in sorted(zip(norm_priors, possible_actions), key=lambda pair: pair[0], reverse=True)]
        norm_priors = sorted(norm_priors, reverse=True)
        return possible_actions, norm_priors
    
    def _optimise_templates(self, possible_actions, priors, optimised_templates):
        """
        Adjust the priors of the possible actions if they are to be optimised.
        
        :param possible_actions: the possible actions
        :param priors: the priors of the possible actions
        :param optimised_templates: the optimised templates
        :return: the updated possible actions and priors
        """
        max_prior = max(priors)
        prior_range = max_prior - min(priors)

        # Assuming optimised_templates is a list of tuples (template, score)
        # where score is a number between 0 and 1
        for index, transformation in enumerate(possible_actions):
            template = transformation.metadata['template']

            for opt_template, score in optimised_templates:
                if template == opt_template:
                    original_prior = priors[index]
                    
                    # Boost factor based on score and prior distribution
                    optimisation_factor = score * prior_range  
                    new_prior = original_prior + optimisation_factor
                    
                    priors[index] = new_prior
                    transformation.metadata['policy_probability'] = new_prior
                    transformation.metadata['classification'] = 'optimised'
                    break 

        norm_priors = [float(i)/sum(priors) for i in priors]
        return possible_actions, norm_priors
            
    def get_actions(
        self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies

        :param molecules: the molecules to consider
        :return: the actions and the priors of those actions
        :raises: PolicyException: if the policy isn't selected
        """
        if not self.selection:
            raise PolicyException("No policy selected")
    
        all_possible_actions = []
        all_priors = []
        
        if self._config.optimise_from_file == True:
            optimised_templates = self._get_optimised_templates(novel=False)
            
            for name in self.selection:
                possible_actions, priors = self[name].get_actions(molecules)
                possible_actions, priors = self._optimise_templates(possible_actions, priors, optimised_templates)
                all_possible_actions.extend(possible_actions)
                all_priors.extend(priors)
                if not self._config.additive_expansion and all_possible_actions:
                    break 
        elif self._config.allow_foreign_templates == True:
            foreign_templates = self.get_foreign_templates()
            
            for name in self.selection:
                compatible_templates = self._check_foreign_compatibility(foreign_templates, molecules)
                
                if len(compatible_templates) != 0:
                    possible_actions, priors = self[name].get_actions(molecules)
                    possible_actions, priors = self._integrate_foreign_templates(possible_actions, priors, foreign_templates, molecules)
                    all_possible_actions.extend(possible_actions)
                    all_priors.extend(priors)
                    if not self._config.additive_expansion and all_possible_actions:
                        break
                else:
                    possible_actions, priors = self[name].get_actions(molecules)
                    all_possible_actions.extend(possible_actions)
                    all_priors.extend(priors)
                    if not self._config.additive_expansion and all_possible_actions:
                        break 
        else:
            for name in self.selection:
                possible_actions, priors = self[name].get_actions(molecules)
                all_possible_actions.extend(possible_actions)
                all_priors.extend(priors)
                if not self._config.additive_expansion and all_possible_actions:
                    break
        return all_possible_actions, all_priors
    
    
    def load(self, source: ExpansionStrategy) -> None:  # type: ignore
            """
            Add a pre-initialized expansion strategy object to the policy

            :param source: the item to add
            """
            if not isinstance(source, ExpansionStrategy):
                raise PolicyException(
                    "Only objects of classes inherited from ExpansionStrategy can be added"
                )
            self._items[source.key] = source

    def load_from_config(self, **config: Any) -> None:
        """
        Load one or more expansion policy from a configuration

        The format should be
        files:
            key:
                - path_to_model
                - path_to_templates
        or
        template-based:
            key:
                - path_to_model
                - path_to_templates
        or
        custom_package.custom_model.CustomClass:
            key:
                param1: value1
                param2: value2

        :param config: the configuration
        """
        files_spec = config.get("files", config.get("template-based", {}))
        for key, policy_spec in files_spec.items():
            modelfile, templatefile = policy_spec
            strategy = TemplateBasedExpansionStrategy(
                key, self._config, source=modelfile, templatefile=templatefile
            )
            self.load(strategy)

        # Load policies specifying a module and class, e.g. package.module.MyStrategyClass
        for strategy_spec, strategy_config in config.items():
            if strategy_spec in ["files", "template-based"]:
                continue
            cls = load_dynamic_class(
                strategy_spec, expansion_strategy_module, PolicyException
            )
            for key, policy_spec in strategy_config.items():
                obj = cls(key, self._config, **(policy_spec or {}))
                self.load(obj)


class FilterPolicy(ContextCollection):
    """
    An abstraction of a filter policy.

    This policy provides a query on a reaction to determine whether it should be rejected

    :param config: the configuration of the tree search
    """

    _collection_name = "filter policy"

    def __init__(self, config: Configuration) -> None:
        super().__init__()
        self._config = config

    def __call__(self, reaction: RetroReaction) -> None:
        return self.apply(reaction)

    def apply(self, reaction: RetroReaction) -> None:
        """
        Apply the all the selected filters on the reaction. If the reaction
        should be rejected a `RejectionException` is raised

        :param reaction: the reaction to filter
        :raises: if the reaction should be rejected or if a policy is selected
        """
        if not self.selection:
            raise PolicyException("No filter policy selected")

        for name in self.selection:
            self[name](reaction)

    def load(self, source: FilterStrategy) -> None:  # type: ignore
        """
        Add a pre-initialized filter strategy object to the policy

        :param source: the item to add
        """
        if not isinstance(source, FilterStrategy):
            raise PolicyException(
                "Only objects of classes inherited from FilterStrategy can be added"
            )
        self._items[source.key] = source

    def load_from_config(self, **config: Any) -> None:
        """
        Load one or more filter policy from a configuration

        The format should be
        files:
            key: path_to_model
        or
        quick-filter:
            key: path_to_model
        or
        custom_package.custom_model.CustomClass:
            key:
                param1: value1
                param2: value2

        :param config: the configuration
        """
        files_spec = config.get("files", config.get("quick-filter", {}))
        for key, modelfile in files_spec.items():
            strategy = QuickKerasFilter(key, self._config, source=modelfile)
            self.load(strategy)

        # Load policies specifying a module and class, e.g. package.module.MyStrategyClass
        for strategy_spec, strategy_config in config.items():
            if strategy_spec in ["files", "quick-filter"]:
                continue
            strategy_spec2 = FILTER_STRATEGY_ALIAS.get(strategy_spec, strategy_spec)
            cls = load_dynamic_class(
                strategy_spec2, filter_strategy_module, PolicyException
            )
            for key, policy_spec in strategy_config.items():
                obj = cls(key, self._config, **(policy_spec or {}))
                self.load(obj)

    
    
            
        
            
            