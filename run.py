import os

import numpy as np

from utils.dataset_lila import DatasetLila, eval_lila

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import json
import logging
import random
import sys

import math
import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer

import utils
from model.model import TargetedTrainingModel
from utils.data_utils import Split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

datasets = {DatasetLila, }

eval_methods = {eval_lila, }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='google/flan-t5-large')
    parser.add_argument('--task_name', type=str, default='GSM8k')
    parser.add_argument('--data_dir', type=str, default='datasets')

    parser.add_argument('--max_seq_length', type=int, default=192)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=9)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=3)
    parser.add_argument('--report_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=5000)
    parser.add_argument('--random_seed', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='t5_results')
    parser.add_argument('--dataset', type=str, default='Lila')
    parser.add_argument('--dataset_multiply', type=str, default='1')
    parser.add_argument('--project_name', type=str, default='targeted_training')
    parser.add_argument('--run_name', type=str, default='flan-t5-large-targeted-training')
    parser.add_argument('--decoder_length', type=int, default=256)
    parser.add_argument('--disable_targeted_training', type=bool, default=False)
    return parser.parse_args()


def model_eval(model, dataloader, args, save_path, data_path, examples, **kwargs):
    results = eval_impl(args, dataloader, model, examples)

    print(f'Output example"')
    for i in random.sample(list(results.keys()), min(10, len(list(results.keys())))):
        print(f'input: {results[i]["input"]}')
        print(f'output: {results[i]["output"]}')
        print(f'uid: {i}')
    print()

    with open(save_path, 'w', encoding='utf-8') as o:
        json.dump(results, o, ensure_ascii=False, indent=4)

    all_dataset_results = {}
    for dataset_name in args.dataset.split(','):
        all_dataset_results |= getattr(sys.modules[__name__], f'eval_{dataset_name.lower()}')(save_path=save_path, data_path=os.path.join(args.data_dir, dataset_name, data_path),
                                                                                              examples=examples, **kwargs)
    logger.info(all_dataset_results)
    return all_dataset_results


def eval_impl(args, dataloader, model, examples):
    model.eval()
    tqdm_bar = tqdm(enumerate(dataloader()), total=dataloader.steps, ncols=125)
    outputs = []
    inputs = []
    uids = []
    with torch.no_grad():
        for idx, (_uids, batch) in tqdm_bar:
            inputs += batch['input_ids'].detach().cpu().numpy().tolist()
            if torch.cuda.device_count() > 1:
                output = model.module.generate(**batch, max_new_tokens=args.decoder_length, do_sample=False)
            else:
                output = model.generate(**batch, max_new_tokens=args.decoder_length, do_sample=False)
            outputs += output
            uids += _uids
            tqdm_bar.set_description(f'Evaluating...')  # avg loss: {round(sum(back_loss) / args.report_steps, 5)}')
    inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    results = dict([(uid, {'input': input, 'output': output}) for uid, input, output in zip(uids, inputs, outputs)])
    return results


def data_collate(features):
    batch = {}
    max_batch_seq_len_src = max([len(f.src_input_ids) for f in features])

    src_input_ids = [f.src_input_ids + [tokenizer.pad_token_id] * (max_batch_seq_len_src - len(f.src_input_ids)) for f in features]
    src_attention_mask = [f.src_attention_mask + [0] * (max_batch_seq_len_src - len(f.src_attention_mask)) for f in features]

    max_batch_seq_len_tgt = max([len(f.tgt_input_ids) if f.tgt_input_ids is not None else 0 for f in features])
    tgt_input_ids = [(f.tgt_input_ids + [-100] * (max_batch_seq_len_tgt - len(f.tgt_input_ids))) if f.tgt_input_ids is not None else [-100] * max_batch_seq_len_tgt for f in
                     features]

    batch['labels'] = torch.tensor(tgt_input_ids, device=device)
    batch['input_ids'] = torch.tensor(src_input_ids, device=device)
    batch['attention_mask'] = torch.tensor(src_attention_mask, device=device)
    example_ids = [f.example_id for f in features]
    batch['example_ids'] = example_ids
    batch['programs'] = [f.program for f in features]
    batch['context'] = [f.context for f in features]

    return example_ids, batch


def main():
    global device
    global tokenizer

    args = parse_args()

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    logging.info(f'args: {args.__dict__}')
    os.environ['RUN_NAME'] = args.run_name
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)  # T5TokenizerFast.from_pretrained(args.model_path)
    utils.config.global_tokenizer = tokenizer

    eval_kwargs = {}  # kwargs for model eval

    train_datasets = []
    dev_datasets = []
    test_datasets = []
    all_datasets = []
    examples = {}

    for dataset_name, dataset_multiply in zip(args.dataset.split(','), args.dataset_multiply.split(',')):
        logger.info(f'LOADING dataset {dataset_name}')

        train_dataset = getattr(sys.modules[__name__], f'Dataset{dataset_name}')(data_dir=os.path.join(args.data_dir, dataset_name), tokenizer=tokenizer,
                                                                                 max_seq_length=args.max_seq_length, mode=Split.train,
                                                                                 disable_targeted_training=args.disable_targeted_training)
        dev_dataset = getattr(sys.modules[__name__], f'Dataset{dataset_name}')(data_dir=os.path.join(args.data_dir, dataset_name), tokenizer=tokenizer,
                                                                               max_seq_length=args.max_seq_length, mode=Split.dev,
                                                                               disable_targeted_training=args.disable_targeted_training)
        test_dataset = getattr(sys.modules[__name__], f'Dataset{dataset_name}')(data_dir=os.path.join(args.data_dir, dataset_name), tokenizer=tokenizer,
                                                                                max_seq_length=args.max_seq_length, mode=Split.test,
                                                                                disable_targeted_training=args.disable_targeted_training)
        train_datasets.append(train_dataset)
        dev_datasets.append(dev_dataset)
        test_datasets.append(test_dataset)
        examples |= dev_dataset.processor.examples
        examples |= test_dataset.processor.examples

    train_datasets = torch.utils.data.ConcatDataset(train_datasets)
    dev_datasets = torch.utils.data.ConcatDataset(dev_datasets)
    test_datasets = torch.utils.data.ConcatDataset(test_datasets)

    wandb.init(project=args.project_name, name=args.run_name)
    logging.info(f'train model on {"gpu" if torch.cuda.is_available() else "cpu"}. GPU num: {torch.cuda.device_count()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TargetedTrainingModel(args.model_path, tokenizer, max_new_tokens=args.decoder_length)

    model.to(device)

    total_step_size = len(train_datasets) * args.epoch // (args.batch_size * args.gradient_accumulation_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_step_size)

    total_steps = 0
    total_idx = 0
    back_loss = []

    best_dev_score = {0.0: None}

    for epoch_idx in range(args.epoch):
        dataloader = TargetedTrainingDataloader(dataset=train_datasets, batch_size=args.batch_size, collate_fn=data_collate, )
        tqdm_bar = tqdm(enumerate(dataloader()), total=dataloader.steps, ncols=125)
        for idx, (uids, batch) in tqdm_bar:
            total_idx += 1
            model.train()

            loss, _ = model(**batch)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if not math.isnan(loss.item()):
                back_loss.append(loss.item())
            if len(back_loss) > args.report_steps * args.gradient_accumulation_steps:
                back_loss = back_loss[1:]

            if len(back_loss) != 0:
                tqdm_bar.set_description(f'Epoch {epoch_idx}, avg loss: {round(sum(back_loss) / len(back_loss), 5)}')

            if total_idx % args.gradient_accumulation_steps == 0 or total_idx == dataloader.steps:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

                total_steps += 1

            if total_idx % args.eval_steps == 0:
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                if not os.path.exists(os.path.join(args.save_path, f'checkpoints_{total_steps}')):
                    os.mkdir(os.path.join(args.save_path, f'checkpoints_{total_steps}'))
                results = {}

                dev_dataloader = TargetedTrainingDataloader(dataset=dev_datasets, batch_size=args.batch_size * 4, collate_fn=data_collate)
                _results = model_eval(model, dev_dataloader, args, os.path.join(args.save_path, f'checkpoints_{total_steps}', 'dev_output.json'), **eval_kwargs,
                                      data_path='dev.json', examples=examples, )

                is_new_best_3 = False
                if _results['mas_acc'] > min(best_dev_score.keys()):
                    is_new_best_3 = True
                    best_dev_score[_results['mas_acc']] = f'checkpoints_{total_steps}'

                    if os.path.exists(os.path.join(args.save_path, f'checkpoints_{total_steps}', 'pytorch_model.bin')):
                        print(f"deleting old checkpoints: " + str(os.path.join(args.save_path, f'checkpoints_{total_steps}', 'pytorch_model.bin')))
                        os.system(f"rm {os.path.join(args.save_path, f'checkpoints_{total_steps}', 'pytorch_model.bin')}")

                    del best_dev_score[min(best_dev_score.keys())]

                for key in _results:
                    results[f'{key}_dev'] = _results[key]

                test_dataloader = TargetedTrainingDataloader(dataset=test_datasets, batch_size=args.batch_size * 4, collate_fn=data_collate)
                _results = model_eval(model, test_dataloader, args, os.path.join(args.save_path, f'checkpoints_{total_steps}', 'test_output.json'), **eval_kwargs,
                                      data_path='test.json', examples=examples, )
                for key in _results:
                    results[f'{key}_test'] = _results[key]

                eval_result = {**results, **{'steps': total_steps, 'lr': lr_scheduler.get_last_lr(), 'epoch': epoch_idx,
                                             'loss': round(sum(back_loss) / len(back_loss) * args.gradient_accumulation_steps, 5)}}
                wandb.log(eval_result)
                logger.warning(eval_result)

                if is_new_best_3:
                    print(f"saving model at {os.path.join(args.save_path, f'checkpoints_{total_steps}', 'pytorch_model.bin')}")
                    torch.save(model, os.path.join(args.save_path, f'checkpoints_{total_steps}', 'pytorch_model.bin'))
            elif total_steps % args.report_steps == 0:
                wandb.log({'steps': total_steps, 'lr': lr_scheduler.get_last_lr(), 'epoch': epoch_idx,
                           'loss': round(sum(back_loss) / (len(back_loss) if len(back_loss) != 0 else 1) * args.gradient_accumulation_steps, 5)})


class TargetedTrainingDataloader:
    def __init__(self, dataset, batch_size, collate_fn, option_datasets=None):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        if option_datasets is None:
            option_datasets = ['MetaLogic', 'ReClor']
        else:
            option_datasets = []
        x = len(list(filter(lambda x: x.split('_')[0] in option_datasets, [dataset[i].example_id for i in range(len(dataset))])))
        y = len(dataset) - x
        # self.steps = max(y - math.ceil(batch_size % 4) * math.ceil(x // (4 * (batch_size // 4))), 0) + math.ceil(x // (4 * (batch_size // 4)))
        self.steps = math.ceil(len(dataset) / batch_size)
        self.features = dict([(self.dataset[i].example_id, self.dataset[i]) for i in range(len(self.dataset))])
        keys = self.features.keys()
        option_dataset_keys = set(filter(lambda x: x.split('_')[0] in option_datasets, keys))
        keys -= option_dataset_keys
        keys = list(keys)
        option_dataset_keys = list(set(map(lambda x: x.split('_positive')[0] if 'positive' in x else x.split('_negative')[0], option_dataset_keys)))
        self.batch_keys = self.generate_batch_keys(keys, option_dataset_keys)

    def keys_dist(self, length, total, min_num, max_num):
        res = [min_num] * length
        total -= min_num * length
        while total > 0:
            loc = random.randint(0, length - 1)
            while res[loc] >= max_num:
                loc = random.randint(0, length - 1)
            res[loc] += 1
            total -= 1
        return res

    def generate_batch_keys(self, keys, option_dataset_keys):
        batch_keys = []
        keys_dist = []
        if len(option_dataset_keys) > 0:
            assert self.batch_size >= 4
            random.shuffle(option_dataset_keys)
            option_keys_dist = self.keys_dist(self.steps, len(option_dataset_keys), 0, self.batch_size // 4)
            for i in range(len(option_keys_dist)):
                _keys = []
                for _ in range(option_keys_dist[i]):
                    key = option_dataset_keys.pop()
                    _keys += [f'{key}_positive', f'{key}_negative_0', f'{key}_negative_1', f'{key}_negative_2']
                batch_keys.append(_keys)
            keys_dist = [self.batch_size - len(batch_keys[i]) for i in range(len(batch_keys))]
        else:
            keys_dist = [self.batch_size for _ in range(self.steps)]
            batch_keys = [[] for _ in range(self.steps)]

        random.shuffle(keys)
        try:
            for i in range(len(keys_dist)):
                temp_key = []
                for _ in range(keys_dist[i]):
                    if len(keys) > 0:
                        temp_key.append((keys.pop()))
                batch_keys[i] += temp_key
        except IndexError:
            logger.info('pop from empty list: keys!')
        return batch_keys

    def __call__(self):
        for batch_keys in self.batch_keys:
            if len(batch_keys) == 0:
                continue
            yield self.collate_fn([self.features[key] for key in batch_keys])


if __name__ == '__main__':
    main()
