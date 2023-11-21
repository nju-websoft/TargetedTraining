import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Union

import math


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    all = "all"
    train_as_dev = "train_as_dev"


@dataclass(frozen=False)
class InputExample:
    example_id: str
    input: str
    output: Optional[str]
    question: Optional[str] = None
    program: Optional[str] = None
    answer: Optional[str] = None
    texts: Union[List[str], Dict[str, str]] = None
    table: Optional[List[List[str]]] = None
    options: Optional[List] = None


@dataclass(frozen=False)
class InputFeatures:
    example_id: str
    src_input_ids: List[int]
    src_attention_mask: List[int]
    context: Union[str, Dict[str, str]]
    tgt_input_ids: Optional[List[int]] = None
    tgt_attention_mask: Optional[List[int]] = None
    program: Optional[str] = None


def program_split(p):
    p = p.replace(';', ',')
    p = re.split(r'[,(]', p)
    p = list(map(str.strip, p))
    p = sum([[_p] if _p[-1] != ')' else [_p[:-1], _p[-1]] for _p in p], [])
    return p


def str_to_num(word):
    scale = 1
    word = word.replace(',', '').strip()
    if '%' in word:
        word = word.replace('%', '')
        scale = 0.01
    try:
        word = int(word) * scale
    except ValueError:
        try:
            word = float(word) * scale
        except ValueError:
            pass
    return None if type(word) == str else word


def eval_program_one_step(it, steps):
    simple_operators = ['surface_cube', 'floor', 'lcm', 'diagonal', 'square_perimeter', 'gcd', 'volume_cylinder',
                        'divide', 'choose', 'triangle_area', 'quadrilateral_area', 'original_price_before_loss',
                        'triangle_perimeter', 'sine', 'cosine', 'log', 'rhombus_area', 'volume_cone', 'circumface',
                        'stream_speed', 'inverse', 'rhombus_perimeter', 'multiply', 'square_edge_by_area', 'add',
                        'volume_rectangular_prism', 'surface_sphere', 'speed', 'reminder', 'tangent',
                        'surface_cylinder', 'volume_sphere', 'power', 'max', 'subtract', 'factorial', 'volume_cube',
                        'square_edge_by_perimeter', 'square_area', 'sqrt', 'triangle_area_three_edges', 'circle_area',
                        'rectangle_area', 'permutation', 'surface_rectangular_prism', 'original_price_before_gain',
                        'cube_edge_by_volume', 'rectangle_perimeter', 'min']
    op, args1 = next(it).split('(')
    op = op.strip()
    args1 = args1.replace(")", '').strip()  # log(2)-->args1=2
    if args1[0] == '#':
        args1 = steps[int(args1[1:])]
    else:
        args1 = args1.replace('const_', '').replace('_', '.')
        args1 = str_to_num(args1)
    if op not in simple_operators:
        return None

    if op == 'surface_cube':
        return 6 * (args1 ** 2)
    # calculate perimeter with known square side length
    if op == 'square_perimeter':
        return 4 * args1
    if op == 'circumface':
        return 2 * getattr(math, 'pi') * args1
    if op == 'inverse':
        return 1.0 / args1
    if op == 'rhombus_perimeter':
        return 4 * args1
    if op == 'square_edge_by_area':
        return getattr(math, 'sqrt')(args1)
    if op == 'surface_sphere':
        return 4 * getattr(math, 'pi') * (args1 ** 2)
    if op == 'volume_sphere':
        return (4 / 3) * getattr(math, 'pi') * (args1 ** 3)
    if op == 'volume_cube':
        return args1 ** 3
    if op == 'square_edge_by_perimeter':
        return args1 / 4
    if op == 'square_area':
        return args1 ** 2
    if op == 'circle_area':
        return getattr(math, 'pi') * args1 ** 2
    if op == 'cube_edge_by_volume':
        return getattr(math, 'pow')(args1, 1 / 3)
    if op in ['floor', 'sine', 'cosine', 'log', 'tangent', 'factorial', 'sqrt']:
        # calculate cube surface area with known cube side length
        if op == 'sine':
            op = 'sin'
        elif op == 'cosine':
            op = 'cos'
        elif op == 'log':
            op = 'log2'
        elif op == 'tangent':
            op = 'tan'
        return getattr(math, op)(args1)

    args2 = next(it).replace(')', '').strip()
    if args2[0] == '#':
        args2 = steps[int(args2[1:])]
    else:
        args2 = args2.replace('const_', '').replace('_', '.')
        args2 = str_to_num(args2)

    if op in ['lcm', 'gcd', 'power']:
        if op == 'power':
            op = 'pow'
        return getattr(math, op)(args1, args2)
    if op == 'diagonal':
        return getattr(math, 'sqrt')(args1 ** 2 + args2 ** 2)
    # calculate the cylinder volume with known base radius and height
    if op == 'volume_cylinder':
        return (getattr(math, 'pi') * args1 ** 2) * args2
    # C_{args1}^{args2}
    if op == 'choose':
        return getattr(math, 'factorial')(args1) / (
                getattr(math, 'factorial')(args1 - args2) * getattr(math, 'factorial')(args2))
    if op in ['triangle_area', 'rhombus_area']:
        return args1 * args2 * 0.5
    # original_price_before_loss(30, 420) = 600
    if op == 'original_price_before_loss':
        return args2 * 100 / (100 - args1)
    if op == 'volume_cone':
        return getattr(math, 'pi') * (args1 ** 2) * args2 / 3.0
    if op == 'stream_speed':
        return 0.5 * (args1 + args2)
    if op == 'speed':
        return args1 / args2
    # reminder(args1, args2) = args1 % args2
    if op == 'reminder':
        return int(args1) % int(args2)
    if op == 'surface_cylinder':
        pi = getattr(math, 'pi')
        return 2 * pi * args1 ** 2 + 2 * pi * args1 * args2
    if op == 'rectangle_area':
        return args1 * args2
    if op == 'permutation':
        return getattr(math, 'factorial')(args1) / getattr(math, 'factorial')(args1 - args2)
    if op == 'original_price_before_gain':
        return args2 * 100 / (100 + args1)
    if op == 'rectangle_perimeter':
        return (args1 + args2) * 2
    if op == 'add':
        return args1 + args2
    if op == 'subtract':
        return args1 - args2
    if op == 'multiply':
        return args1 * args2
    if op == 'divide':
        return args1 / args2
    if op == 'max':
        return max(args1, args2)
    if op == 'min':
        return min(args1, args2)

    args3 = next(it).replace(')', '').strip()
    if args3[0] == '#':
        args3 = steps[int(args3[1:])]
    else:
        args3 = args3.replace('const_', '').replace('_', '.')
        args3 = str_to_num(args3)

    # quadrilateral_area(h,a,b) 0.5*h*(a+b)
    if op == 'quadrilateral_area':
        return 0.5 * args1 * (args2 + args3)
    # triangle_perimeter(a, b, c) a+b+c
    if op == 'triangle_perimeter':
        return args1 + args2 + args3
    if op == 'volume_rectangular_prism':
        return args1 * args2 * args3
    if op == 'triangle_area_three_edges':
        p = (args1 + args2 + args3) / 2
        return getattr(math, 'sqrt')(p * (p - args1) * (p - args2) * (p - args3))
    if op == 'surface_rectangular_prism':
        return 2 * (args1 * args2 + args1 * args3 + args2 * args3)
    return getattr(sys.modules[__name__], op)(args1, args2, args3)


def handle_scale(program: List[str]):
    scale = 1
    scale_map = {'thousand': 1000, 'million': 1000000, 'billion': 1000000000, 'percent': 0.01}
    for p_index, p in enumerate(program):
        if 'scale' in p:
            scale_s = p.replace('scale', '').replace('(', '').replace(')', '')
            scale = scale_map[scale_s] if scale_s in scale_map else 1
            program[p_index] = None
    program = list(filter(lambda x: x is not None, program))
    return scale, program


def eval_program(program):
    program = program.replace(';', ',')
    program = program.split(',')
    scale, program = handle_scale(program)
    it = iter(program)
    steps = []
    try:
        while True:
            res = eval_program_one_step(it, steps)
            if res is None:
                raise StopIteration
            steps.append(res)
    except (StopIteration, ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError):
        if len(steps) > 0:
            return steps[-1] * scale
        else:
            return None


class ProcessorDataset:

    def __init__(self, **kwargs):
        self.examples = {}

    def get_train_examples(self, **kwargs):
        raise NotImplementedError()

    def get_dev_examples(self, **kwargs):
        raise NotImplementedError()

    def get_test_examples(self, **kwargs):
        raise NotImplementedError()

    def _read_json(self, **kwargs):
        raise NotImplementedError()

    def _create_examples(self, **kwargs) -> List[InputExample]:
        raise NotImplementedError()


def generate_expression_question(question, context=None):
    return f'''{question} \nContext: {context}'''
