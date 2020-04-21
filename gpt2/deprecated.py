
    '''
    #this was inside :def tp_loss() @ run_gpt2.py
    #this is duplicative. not proper to perform
    def update_cls_indices(indices, premise_lengths, nchoices=0):
        premise_lengths = torch.cuda.LongTensor(premise_lengths)
        premise_lengths +=1 # include [sep] token

        if indices.shape[-1] == 3 and nchoice>=1: # for cls_fals_indices
            premise_lengths = premise_lengths.unsqueeze(-1).expand(-1,nchoices).flatten()
            t_zero = torch.zeros_like(premise_lengths)
            adding = torch.stack((t_zero, t_zero, premise_lengths)).t()
            indices = indices + adding
            return indices
        elif indices.shape[-1] == 2 and nchoice==0: # for cls_tru_indices
            t_zero = torch.zeros_like(premise_lengths)
            adding = torch.stack( (t_zero, premise_lengths) ).t()
            indices = indices + adding
            return indices

        else:
            assert False, "indices has problem"'''


    #indices update: canceled
    #cls_tru_indices = update_cls_indices(cls_tru_indices, premise_lengths)
    #cls_fals_indices = update_cls_indices(cls_tru_indices, premise_lengths, nchoices=n_choice)
