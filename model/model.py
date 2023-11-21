import logging

from torch import nn
from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Config

logger = logging.getLogger(__name__)


class TargetedTrainingModel(nn.Module):
    def __init__(self, model_path, tokenizer: PreTrainedTokenizer, max_new_tokens: int,):
        super().__init__()
        logger.info(f'loading model from {model_path}...')
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.config = T5Config.from_pretrained(model_path)
        logger.info(f'loading model DONE!')
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def forward(self, **kwargs):
        kwargs['program_ids'] = kwargs['labels']
        labels = kwargs.pop('labels')
        batch = {'input_ids': kwargs['input_ids'], 'attention_mask': kwargs.pop('attention_mask'), 'labels': labels}
        t5_output = self.t5_model(**batch)
        t5_loss = t5_output.loss

        return t5_loss, t5_output.logits

    def generate(self, **kwargs):
        if 'labels' in kwargs:
            del kwargs['labels']
        batch = {'input_ids': kwargs['input_ids'], 'attention_mask': kwargs.pop('attention_mask'), }
        if 'programs' in kwargs:
            del kwargs['programs']
        programs = self.t5_model.generate(**batch, max_new_tokens=self.max_new_tokens)
        programs = self.tokenizer.batch_decode(programs, skip_special_tokens=True)

        results = [{'program': program, } for program in programs]
        return results
