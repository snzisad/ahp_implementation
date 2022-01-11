import json
# from pyahp import parse


from queue import Queue
import numpy as np

from AHP.errors import AHPModelError
from AHP.AHPCriterion import AHPCriterion
from AHP.utils import normalize_priorities
from AHP.methods import ApproximateMethod, EigenvalueMethod, GeometricMethod


class AHPModel:
    """AHPModel

    Args:
        model (dict): The Analytic Hierarchy Process model.
        solver (pyahp.methods): Method used when calculating the priorities of the lower layer.
    """

    def __init__(self, model, solver=EigenvalueMethod):
        self.solver = solver()
        self.preference_matrices = model['preferenceMatrices']

        criteria = model.get('criteria')
        self.criteria = [AHPCriterion(n, model, solver=solver) for n in criteria]

    def get_priorities(self, round_results=True, decimals=3):
        """Get the priority of the nodes in the level below this node.

        Args:
            round_results (bool): Return rounded priorities. Default is True.
            decimals (int): Number of decimals to round to, ignored if `round_results=False`. Default is 3.

        Returns:
            Global priorities of the alternatives in the model, rounded to `decimals` positions if `round_results=True`.
        """
        crit_pm = np.array(self.preference_matrices['criteria'])
        crit_pr = self.solver.estimate(crit_pm)
        print("Criteria Weights: ")
        print(crit_pr)

        # print("\nSub Criteria Weights: ")
        # for s in self.criteria:
        #     print(s.sub_criteria)
        sub_priorities = []
        alter_priorities = []

        for s in self.criteria:
            alter_priorities.append(s)
            if not s.leaf:
                sub_priorities.append(s)

        if len(sub_priorities)>0:
            print("\nSub Criteria Weights: ")
            for s in sub_priorities:
                p_m = np.array(s.preference_matrices[s.p_m_key])
                sub_crit_pr = s.solver.estimate(p_m)
                print(sub_crit_pr)

        print("\nAlternate Weights: ")
        for s in alter_priorities:
            print(s.get_priorities())

        priorities = normalize_priorities(self.criteria, crit_pr)

        if round_results:
            return np.around(priorities, decimals=decimals)

        return priorities


def _check_ahp_list(name, value):
    if not isinstance(value, list):
        raise AHPModelError('Expecting {} to be a list got {}'.format(name, _type(value)))

    if not value:
        raise AHPModelError('{} list empty'.format(name))

    for elem in value:
        if not isinstance(elem, str):
            raise AHPModelError('Expecting {} list to have string got {}'.format(name, _type(elem)))

    if len(value) != len(set(value)):
        raise AHPModelError('{} list contains duplicates'.format(name))


def _check_ahp_preference_matrix(name, p_m, kind, length):
    if p_m is None:
        raise AHPModelError('Missing {} preference matrix for {}'.format(kind, name))

    p_m = np.array(p_m)

    width, height = p_m.shape
    if width != height or width != length:
        raise AHPModelError(
            'Expecting {0}:{1} preference matrix to be {2}x{2} got {3}x{4}'.format(kind,
                                                                                   name,
                                                                                   length,
                                                                                   width,
                                                                                   height)
        )


def validate_model(model):
    """Validate the passed AHP model.

    Args:
        model (dict): The Analytic Hierarchy Process model.

    Raises:
        AHPModelError when the model validation fails.
    """

    if not isinstance(model, dict):
        raise AHPModelError('Expecting a config dictionary got {}'.format(_type(model)))

    method = model['method']
    if not isinstance(method, str):
        raise AHPModelError('Expecting method to be string got {}'.format(_type(method)))

    if method not in ['approximate', 'eigenvalue', 'geometric']:
        raise AHPModelError('Expecting method to be approximate, eigenvalue or geometric')

    _check_ahp_list('criteria', model['criteria'])
    _check_ahp_list('alternatives', model['alternatives'])

    n_alternatives = len(model['alternatives'])
    preference_matrices = model['preferenceMatrices']
    criteria = model['criteria']
    criteria_queue = Queue()

    criteria_p_m = preference_matrices.get('criteria')
    _check_ahp_preference_matrix(name='criteria',
                                 p_m=criteria_p_m,
                                 kind="criteria",
                                 length=len(criteria))

    for criterion in criteria:
        criteria_queue.put(criterion)
        
    sub_criteria_map = model.get('subCriteria')

    while not criteria_queue.empty():
        criterion = criteria_queue.get()
        sub_criteria = sub_criteria_map.get(criterion)
        
        if sub_criteria:
            _check_ahp_list('subCriteria:{}'.format(criterion), sub_criteria)

            p_m = preference_matrices.get('subCriteria:{}'.format(criterion))
            _check_ahp_preference_matrix(name=criterion,
                                         p_m=p_m,
                                         kind="subCriteria",
                                         length=len(sub_criteria))

            for sub_criterion in sub_criteria:
                criteria_queue.put(sub_criterion)
        else:
            p_m = preference_matrices.get('alternatives:{}'.format(criterion))
            _check_ahp_preference_matrix(name=criterion,
                                         p_m=p_m,
                                         kind="alternatives",
                                         length=n_alternatives)


def parse(model):
    """Parse the passed AHP model.

    Args:
        model (dict): The Analytic Hierarchy Process model.

    Returns:
        AHPModel with the specified solver.
    """
    validate_model(model)

    method = model['method']
    solver = EigenvalueMethod

    if method == 'approximate':
        solver = ApproximateMethod
    elif method == 'geometric':
        solver = GeometricMethod

    return AHPModel(model, solver)
