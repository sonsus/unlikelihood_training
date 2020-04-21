import torch
import jsonlines as jsl
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import List

def init_special_tokens(tokenizer):
    #add special tokens to tokenizer and return it (returns updated tokenizer)
    #keys must be in [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``].
    specials = {
        'pad_token': '[PAD]',
        #'unk_token': '[UNK]', #not in use for byte level BPE like in GPT
        'cls_token': '[CLS]',
        'sep_token': '[SEP]',
        'eos_token': '[EOS]', # don't need it but just in case we need it later
        'bos_token': '[BOS]',
        #no mask token for gpt2
    }
    tokenizer.add_special_tokens(specials)

    #check it worked
    assert tokenizer.pad_token== '[PAD]'
    #assert tokenizer.unk_token== '[UNK]'
    assert tokenizer.cls_token== '[CLS]'
    assert tokenizer.sep_token== '[SEP]'
    assert tokenizer.eos_token== '[EOS]'
    assert tokenizer.bos_token== '[BOS]'

    return tokenizer



'''dataloader always need to load things w/o special tokens

only when needed, add special tokens with provided functions below
'''

#lets add cls and sep tokens first, and then remove it for pretraining
def concat_process(text1, text2, tokenizer, mode):
    if mode == 'relFT': #looks like it never used
        return f"{tokenizer.bos_token} {text1} {tokenizer.sep_token} {text2} {tokenizer.cls_token}"
    elif mode == 'CLM':
        return f"{text1} {text2}"


class Example:
    '''
    Example class attributes
        .premise : str
        .true_h : str

        .pre_tru : tensor
        .pre_tru_id: tensor (premise, hypotheses label for transformer architecture)

        .n_false : int
        [.pre_fals : tensor]
        [.pre_fals_id: tensor]
    '''
    def __init__(self, obj, tokenizer, args): # mode in ["CLM", "relFT"]
        self.args = args
        self.premise = obj['premise']
        self.true_h = obj['true_h']
        self.n_false = 0 # default
        self.false_hs = obj['false_hs']
        self.tokenizer = tokenizer

        #concat
        #self.pre_tru, self.pre_tru_id = tokenizer.encode_plus( gpt_build_with_specials(self.premise, self.true_h, tokenizer), add_special_tokens=True, return_token_type_ids=True)
        self.pre_tru = self.tokenizer.encode_plus( concat_process(self.premise, self.true_h, self.tokenizer, mode= 'CLM') )['input_ids']
        self.pre_tru_add = self.tokenizer.encode_plus( concat_process(self.premise, self.true_h, self.tokenizer, mode= 'relFT') )['input_ids'] if args.input_mode=='relFT' else []

        self.premise_token_ids = self.tokenizer.encode_plus(self.premise)
        self.premise_length = len(self.premise_token_ids)

        self.premise_token_ids.append(self.tokenizer.sep_token_id)#add this after defining length so that we don't need to modify the length

        if obj['false_hs']:
            if obj['false_hs'][0]: # to avoid index error, checked stepwise
                #tups_list = [tokenizer(self.premise, n, add_special_tokens=True, return_token_type_ids=True) for n in obj['false_hs'] ]

                self.pre_fals = [self.tokenizer.encode_plus( concat_process(self.premise, n, self.tokenizer, mode= 'CLM')['input_ids']) for n in obj['false_hs'] ]
                self.pre_fals_add = [self.tokenizer.encode_plus( concat_process(self.premise, n, self.tokenizer, mode= 'relFT')['input_ids']) for n in obj['false_hs'] if args.input_mode=='relFT' ]
                #self.pre_fals_id = [tup[1] for tup in tups_list]
                self.n_false = len(self.pre_fals)
                if self.n_false > 1:
                    self.pre_fals = pad_sequence(self.pre_fals, batch_first=True, padding_value=int(self.tokenizer.pad_token_id))
                    #self.pre_fals_id = pad_sequence(self.pre_fals_id, batch_first=True, padding_value=1) #token_type_ids should be 1 for continuation


class Batch:
    '''
    Batch class attributes
        .premises : List[str]
        .true_hs : List[str]
        .false_hs_s : List[List[str]]
        .pre_tru, : 2d tensor (bsz, maxlen)                     //.pre_tru_id
        .pre_fals, : 3d tensor (bsz, n_false, maxlen)           //.pre_fals_id
    '''
    def __init__(self, listofexamples, tokenizer):
        self.batch_size = len(listofexamples)
        self.n_false = listofexamples[0].n_false
        self.tokenizer = tokenizer
        #this only works with negative example present dataset
        #assert self.n_false >= 1, f"self.n_false needs to be larger or eq to 1, now {self.n_false}"

        def to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(torch.device('cuda'))
        def longtensors(obj:List[List[int]])->List[torch.Tensor]:
            return [torch.Tensor(l).long() if l else self.tokenizer.pad_token_id*torch.ones(1).long() for l in obj]

        def longtensors_wdevice(obj:List[List[int]])->List[torch.cuda.Tensor]:
            return [to_device(torch.Tensor(l).long() if l else self.tokenizer.pad_token_id*torch.ones(1).long()) for l in obj]

        def make_it_batched(listofexamples, tokenizer):
            #pre_tru, pre_tru_id, pre_fals, pre_fals_id = [], [], [], []
            pre_tru, pre_fals = [], [],
            pre_tru_add, pre_fals_add = [], []
            premises, true_hs, false_hs_s = [], [], []
            premises_tensors, premises_lengths = [], []
            for ex in listofexamples:
                pre_tru.append(ex.pre_tru)
                #pre_tru_id.append(ex.pre_tru_id)
                if self.n_false>0:
                    pre_fals.append(ex.pre_fals)
                    #pre_fals_id.append(ex.pre_fals_id)
                premises.append(ex.premise)
                true_hs.append(ex.true_h)

                premises_tensors.append(ex.premise_token_ids)
                premises_lengths.append(ex.premise_length)

                if self.n_false>0:
                    false_hs_s.append(ex.false_hs)

                if ex.pre_tru_add and ex.pre_fals_add: # when special tokens added encoded tokens are
                    pre_tru_add.append(ex.pre_tru_add)
                    pre_fals_add.append(ex.pre_fals_add)

            pre_tru, pre_fals = longtensors(pre_tru), longtensors(pre_fals)
            premises_tensors = longtensors_wdevice(premises_tensors)
            pre_tru_add = longtensors(pre_tru_add)
            pre_fals_add = longtensors(pre_fals_add)

            pre_tru = pad_sequence(pre_tru, batch_first=True, padding_value=tokenizer.pad_token_id)
            #pre_tru_id = pad_sequence(pre_tru_id, batch_first=True, padding_value=1)
            if self.n_false>0:
                pre_fals = pad_sequence(pre_fals, batch_first=True, padding_value=tokenizer.pad_token_id)
            #pre_fals_id = pad_sequence(pre_fals_id, batch_first=True, padding_value=1)

            pre_tru_add = pad_sequence(pre_tru_add, batch_first=True, padding_value=tokenizer.pad_token_id)
            pre_fals_add = pad_sequence(pre_fals_add, batch_first=True, padding_value=tokenizer.pad_token_id)

            #return pre_tru, pre_tru_id, pre_fals, pre_fals_id, premises, true_hs, false_hs_s
            return to_device(pre_tru), to_device(pre_fals), to_device(pre_tru_add), to_device(pre_fals_add), premises_tensors, premises_lengths, premises, true_hs, false_hs_s

        #self.pre_tru, self.pre_tru_id, self.pre_fals, self.pre_fals_id, \
        self.pre_tru,  self.pre_fals,  \
        self.pre_tru_add, self.pre_fals_add, \
        self.premises_tensors, self.premises_lengths, \
        self.premises, self.true_hs, self.false_hs_s = make_it_batched(listofexamples, tokenizer)

        assert len(self.pre_tru) == self.batch_size
        assert len(self.premises) == self.batch_size
        if self.n_false > 0:
            assert len(self.pre_fals) == self.batch_size
            assert self.pre_fals.size()[1] == self.n_false


class PosNegDataset(Dataset):
    def __init__(self, args, tokenizer, split='choose_among_train_val_test'): # args.data_path has train/val/test.jsonl
        self.args = args
        self.split = split
        self._ds = []
        #self.tokenizer = tokenizer

        self.root = Path(args.data_path)
        with jsl.open(self.root / f"{self.split}.jsonl") as reader:
            reader = list(reader)#now tqdm shows pbar properly
            for obj in tqdm(reader):
                ex = Example(obj, tokenizer, args)
                self._ds.append(ex)

    def __getitem__(self, index):
        return self._ds[index]

    def __len__(self):
        return len(self._ds)


def collate(listofexamples):
    '''
    in
        listofexamples: List[Example]
    out
        collated ex: Batch
    '''
    tokenizer = listofexamples[0].tokenizer
    #batch.pre_tru.shape : bsz, max_len
    #batch.pre_fals.shape : bsz, n_false, max_len
    return Batch(listofexamples, tokenizer)


def get_dataloaders(args, tokenizer, spl='train or val or test'):

    ds = PosNegDataset(args, tokenizer, split= spl)
    loader = DataLoader(dataset = ds,
                        batch_size = args.batch_size,
                        shuffle = (spl == 'train'),
                        num_workers = args.num_workers,
                        collate_fn = collate
                    )
    return loader
