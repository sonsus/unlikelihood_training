import os
import logging

from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboard import default, program
import tensorflow as tf
from typing import Dict, List
from utils import *
from ipdb import set_trace


class Logger:
    def __init__(self, args, path=None):
        #self.log_cmd = args.log_cmd
        if path is not None: # when running evaluate
            if args.embedplt:
                logdir = Path(path) / f"embedviz_{args.nlu_path}_{get_now()}"
            else:
                logdir = Path(path) / f"{args.training_mode}.{get_now()}"

        else: # when running train
            log_name = get_dirname_from_args(args) + f"_{get_now()}"
            logdir = str( Path(args.log_path) / log_name)
            if str(args.nlg_path)==str(args.nlu_path) and str(args.nlg_path)=="DBG":
                logdir = 'log/DBG'
        Path(logdir).mkdir(parents=True, exist_ok=True)
        args.log_path = logdir
        self.tfboard = SummaryWriter(str(logdir))

    def __call__(self, name, val, n_iter):
        self.tfboard.add_scalar(name, val, n_iter)
        #if self.log_cmd:
        #    tqdm.write(f'{n_iter}:({name},{val})')



# forward compatibility for version > 1.12
def get_assets_zip_provider():
  """Opens stock TensorBoard web assets collection.
  Returns:
    Returns function that returns a newly opened file handle to zip file
    containing static assets for stock TensorBoard, or None if webfiles.zip
    could not be found. The value the callback returns must be closed. The
    paths inside the zip file are considered absolute paths on the web server.
  """
  path = os.path.join(tf.resource_loader.get_data_files_path(), 'webfiles.zip')
  if not os.path.exists(path):
        print('webfiles.zip static assets not found: %s', path)
        return None
  return lambda: open(path, 'rb')

def log_embeddings(logger, embeddings, meta, metaheader=None, step=0, tag='default'):
    # logger: logger obj
    # sentences: list of sentences (length = N)
    # embeddings: N*D tensor
    # pos_or_neg: expects 'p(ositive)' or 'n(egative)' or 'g(enerated)

    logger.tfboard.add_embedding(embeddings,
                                metadata=meta,
                                tag=tag,
                                metadata_header=metaheader,
                                global_step=step
                                )

def log_results(logger, name, state, step):
    for key, val in state.metrics.items():
        if isinstance(val, dict):
            for key2, v in val.items():
                logger(f"{name}/{key}/{key2}" , v, step)
        else:
            logger(f"{name}/{key}" , val, step)

def log_lr(logger, name, optimizer, ep):
    lr=0;
    for param_group in optimizer.param_groups:
        lr= param_group['lr']
        break

    logger(f"{name}", lr, ep)

def log_args(args):
    logdir = Path(args.log_path)
    #logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / "args.json").open(mode='w') as f:
        json.dump(str(args), f, indent=4 )
    print("args logged! @ " + f"{logdir}/args.json")

def log_text(args, ep=0, step=0, vocab=None, src=None, gen=None, trg= None, name= None): # gen: required!
    if args.savestep: #if step is given, ep isnt given
        ep = step #below works just same as step
        assert step #step > 0

    teacher = trg is not None

    logdir = Path(args.log_path)

    if src is not None:
        srcsents = to_string(vocab, src)

    gensents = to_string(vocab, gen) if args.beamsize==1 or args._training else to_string2(vocab, gen)

    logfile = (logdir / f"ep{ep}_{name}_src.txt") if not args.savestep else (logdir / f"step{ep}_{name}_src.txt")
    logfile2 = (logdir / f"ep{ep}_{name}_gen_beam{args.beamsize}.txt") if not args.savestep else (logdir / f"step{ep}_{name}_gen_beam{args.beamsize}.txt")

    #logfile2 = logdir / f"ep{ep}_{name}_val_gen.txt" if args.beamsize==1 or args._training else logdir/f"ep{ep}_{name}_val_beam{args.beamsize}.txt"

    if teacher:
        trgsents = to_string(vocab, trg)
        logfile1 = (logdir / f"ep{ep}_{name}_trg.txt") if not args.savestep else (logdir/f"step{ep}_{name}_trg.txt")

        with logfile.open(mode='w', errors='replace') as f, logfile1.open(mode='w', errors='replace') as f1, logfile2.open(mode="w", errors='replace') as f2:
            for i, gen in enumerate(gensents):
                if src is not None:
                    f.write(f"{i}:  {rep(srcsents[i])}\n")
                if trg is not None:
                    f1.write(f"{i}: {rep(trgsents[i])}\n")
                f2.write(f"{i}:  {rep(gensents[i])}\n")
        print(f"wrote to \n\t{logfile},\n\t{logfile1},\n\t{logfile2}")


    else:
        with logfile.open(mode='w', errors='replace') as f, logfile2.open(mode="w", errors='replace') as f2:
            for i, gen in enumerate(gensents):
                if src is not None:
                    f.write(f"{i}:  {rep(srcsents[i])}\n")
                f2.write(f"{i}:  {rep(gensents[i])}\n")

        print(f"wrote to \n\t{logfile},\n\t{logfile2}")



def get_logger(args, path=None):
    return Logger(args, path=path)





'''def run_tensorboard(log_path):
    log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
    log = logging.getLogger('tensorflow').setLevel(logging.ERROR)

    port_num = abs(hash(log_path)) % (8800) + 1025  # above 1024, below 10000
    tb = program.TensorBoard(default.get_plugins(), get_assets_zip_provider())
    tb.configure(argv=[None, '--logdir', str(log_path), '--port', str(port_num),
                       '--samples_per_plugin', 'text=100'])
    url = tb.launch()
    return url
'''


'''def log_txt(logger:Logger,
    tag:str,
    srcstr:List[str],
    genstr:List[str],
    step, logfirstonly = True) -> None:

    if logfirstonly:
        srcstrs = srcstr[0]
        genstrs = genstr[0]
        logger.tfboard.add_text(f'{tag}_input', srcstrs, step)
        logger.tfboard.add_text(f'{tag}_gen', genstrs, step)
        else: #log all in the batch
        for i, (srctxt, gentxt) in enumerate(zip(srcstr, genstr)):
            logger.tfboard.add_text(f'{tag}_input', f"{i+step}: {srctxt}", step)
            logger.tfboard.add_text(f'{tag}_gen', f"{i+step}: {gentxt}", step)'''
