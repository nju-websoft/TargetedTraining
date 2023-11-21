import json
import logging
import os
from typing import List

import nltk
from filelock import FileLock
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from utils.data_utils import InputExample, InputFeatures, Split, ProcessorDataset, generate_expression_question, eval_program, str_to_num

from utils.unifiedreasoner_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ProcessorLila(ProcessorDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disable_targeted_training = kwargs['disable_targeted_training']

    def get_train_examples(self, tokenizer, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        _examples = self._create_examples(tokenizer, f'{data_dir}/train.json', "train")
        self.examples |= dict([('Lila_' + e.example_id, e) for e in _examples])
        return _examples

    def get_dev_examples(self, tokenizer, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        _examples = self._create_examples(tokenizer, f'{data_dir}/dev.json', "dev")
        self.examples |= dict([('Lila_' + e.example_id, e) for e in _examples])
        return _examples

    def get_test_examples(self, tokenizer, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        _examples = self._create_examples(tokenizer, f'{data_dir}/test.json', "test")
        self.examples |= dict([('Lila_' + e.example_id, e) for e in _examples])
        return _examples

    def _read_json(self, input_dir, data_type):
        datas = json.load(open(input_dir, 'r', encoding='utf-8'))
        return datas

    def load_sift_refine_datas_with_loss(self, file_path, top_ratio=0.8):
        datas = json.load(open(file_path, 'r', encoding='utf-8'))
        refine_program_datas = datas['refined_programs']
        easy_questions = datas['easy_questions']

        all_refine_programs_loss = sum([refine_program_datas[key]['loss'] for key in refine_program_datas], [])
        all_easy_question_loss = []
        for key in easy_questions:
            for idx in range(len(easy_questions[key])):
                all_easy_question_loss.append(easy_questions[key][idx]['loss'])

        refine_programs_loss_bar = sorted(all_refine_programs_loss)[int(top_ratio * len(all_refine_programs_loss))]
        easy_question_loss_bar = sorted(all_easy_question_loss)[int(top_ratio * len(all_easy_question_loss))]

        filterd_refine_programs = {}
        filterd_easy_questions = {}

        for key in refine_program_datas:
            filterd_refine_programs[key] = []
            for program, loss in zip(refine_program_datas[key]['programs'], refine_program_datas[key]['loss']):
                if loss < refine_programs_loss_bar:
                    filterd_refine_programs[key].append(program)

        for key in easy_questions:
            filterd_easy_questions[key] = []
            for question_program in easy_questions[key]:
                if question_program['loss'] < easy_question_loss_bar:
                    filterd_easy_questions[key].append({'question': question_program['question'], 'program': question_program['program']})

        return filterd_refine_programs, filterd_easy_questions

    def _create_examples(self, tokenizer, data_dir, data_type) -> List[InputExample]:
        examples = []
        raw_datas = self._read_json(data_dir, data_type)
        refine_programs, easy_questions = self.load_sift_refine_datas_with_loss(f'datasets/Lila/sift_refine_datas.json')

        origin_datas_repeat_times = 2

        use_new_program_count = 0
        use_easy_question_count = 0
        skipping_count = 0

        if data_type == 'train' and not self.disable_targeted_training:
            for example_id in easy_questions:
                datas = easy_questions[example_id]
                for idx, data in enumerate(datas):
                    question = data['question']
                    program = data['program']
                    *texts, question = nltk.sent_tokenize(question)

                    examples.append(InputExample(example_id=example_id + f'_easy_{idx}',
                                                 input=generate_expression_question(question, ' '.join(texts)),
                                                 output=re_program(program),
                                                 texts=texts,
                                                 question=question,
                                                 program=program, ))
                    use_easy_question_count += 1
                if len(datas) == 0:
                    skipping_count += 1

        for data_index, example_id in tqdm(enumerate(raw_datas), total=len(raw_datas), desc=f'Creating {data_type} examples', ncols=125):
            data = raw_datas[example_id]
            question = data['question']
            program = data['program']
            answer = data['answer']
            *texts, question = nltk.sent_tokenize(question)

            use_new_program = False
            if not self.disable_targeted_training and example_id in refine_programs:
                assert data_type == 'train'
                for prog_idx, program in enumerate(refine_programs[example_id]):
                    examples.append(InputExample(example_id=f'{example_id}_{prog_idx}',
                                                 input=generate_expression_question(question, ' '.join(texts)),
                                                 output=re_program(program),
                                                 texts=texts,
                                                 question=question,
                                                 answer=answer,
                                                 program=program, ))
                    use_new_program_count += 1
                    use_new_program = True

            if not use_new_program:  # use origin program
                examples.append(InputExample(example_id=example_id,
                                             input=generate_expression_question(question, ' '.join(texts)),
                                             output=re_program(program),
                                             texts=texts,
                                             question=question,
                                             answer=answer,
                                             program=program, ))

                if not self.disable_targeted_training and data_type == 'train':  # duplicate to balance training data
                    for i in range(origin_datas_repeat_times - 1):
                        examples.append(InputExample(example_id=example_id + f'_dup_{i}',
                                                     input=generate_expression_question(question, ' '.join(texts)),
                                                     output=re_program(program),
                                                     texts=texts,
                                                     question=question,
                                                     answer=answer,
                                                     program=program, ))

        print(f'use_new_program_count: {use_new_program_count}')
        print(f'skipping_count: {skipping_count}')
        print(f'use_easy_question_count: {use_easy_question_count}')
        return examples


class DatasetLila(BaseDataset):
    features: List[InputFeatures]

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, disable_targeted_training: bool = False, data_dir='dataset/Lila', mode: Split = Split.train, ):
        super().__init__()
        self.processor = ProcessorLila(disable_targeted_training=disable_targeted_training)

        if self.cache(data_dir, mode, tokenizer.__class__.__name__, max_seq_length) == 'load_from_cache':
            return

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        with FileLock(f'.tmp_filelock.lock'):
            logger.info(f"Creating features from dataset file at {data_dir}")
            if mode == Split.dev:
                examples = self.processor.get_dev_examples(tokenizer, data_dir)
            elif mode == Split.test:
                examples = self.processor.get_test_examples(tokenizer, data_dir)
            elif mode == Split.train:
                examples = self.processor.get_train_examples(tokenizer, data_dir)
            else:
                raise NotImplementedError
            logger.info(f"{mode} examples: {len(examples)}")
            self.features = convert_examples_to_features_lila(examples, tokenizer=tokenizer, max_length=max_seq_length, mode=mode)
            self.cache(data_dir, mode, tokenizer.__class__.__name__, max_seq_length)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def re_program(program):
    operators = ["add", "divide", "exp", "greater", "multiply", "subtract"]
    for operator in operators:
        program = program.replace(f'),{operator}', f');{operator}').replace(f'), {operator}', f'); {operator}')
    return program


def convert_examples_to_features_lila(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        mode=Split.train,
) -> List[InputFeatures]:
    features = []
    t = tqdm(enumerate(examples), total=len(examples), desc=f'Converting {mode} examples to features', ncols=125)
    truncation_count = 0
    max_data_length = 0
    for data_index, example in t:
        src = tokenizer(example.input, truncation=True, max_length=max_length)
        tgt = tokenizer(example.output, truncation=True, max_length=max_length)

        features.append(InputFeatures(example_id=f'Lila_{example.example_id}',
                                      src_input_ids=src.input_ids, src_attention_mask=src.attention_mask,
                                      tgt_input_ids=tgt.input_ids, tgt_attention_mask=tgt.attention_mask,
                                      context=' '.join(example.texts) if example.texts is not None else None))
        max_data_length = max(max_data_length, max(len(src.input_ids), len(tgt.input_ids)))
        if len(src.input_ids) == max_length or len(tgt.input_ids) == max_length:
            truncation_count += 1
            if mode == Split.train:
                continue
    print(f'truncation_count: {truncation_count}, max_data_length: {max_data_length}')
    return features


def eval_lila(save_path, data_path, **kwargs):
    preds = json.load(open(save_path, 'r', encoding='utf-8'))
    origin_datas = json.load(open(os.path.dirname(data_path) + '/train.json', 'r', encoding='utf-8')) | \
                   json.load(open(os.path.dirname(data_path) + '/dev.json', 'r', encoding='utf-8')) | \
                   json.load(open(os.path.dirname(data_path) + '/test.json', 'r', encoding='utf-8'))

    total_count = 0
    correct_count = 0
    dataset_count = {}
    dataset_count['mas'] = {'total_count': 0, 'correct_count': 0}
    for key in preds:
        if not key.startswith('Lila_'):
            continue
        dataset_name = '_'.join(key.replace('Lila_', '').split('_')[:-1])
        if dataset_name not in dataset_count:
            dataset_count[dataset_name] = {'total_count': 0, 'correct_count': 0}
        program = preds[key]['output']['program']
        result = eval_program(program) if str_to_num(program) is None else program.strip()
        answer = origin_datas[key.replace('Lila_', '')]['answer'][0]
        try:
            if result is not None and round(float(result), 5) == round(float(answer), 5):
                correct_count += 1
                dataset_count[dataset_name]['correct_count'] += 1

        except ValueError:
            pass
        except OverflowError:
            pass
        total_count += 1
        dataset_count[dataset_name]['total_count'] += 1

    metrics = {'acc': round(correct_count / total_count, 5)}
    for dataset_name in dataset_count:
        if dataset_count[dataset_name]['total_count'] == 0:
            continue
        metrics |= {f'{dataset_name}_acc': round(dataset_count[dataset_name]['correct_count'] / dataset_count[dataset_name]['total_count'], 5)}
    return metrics
