import argparse
import torch
from pytorch_lightning import seed_everything
from transformers import logging

from utils.utils import add_program_args, get_datamodule

from utils.pylogger import get_pylogger

torch.multiprocessing.set_sharing_strategy('file_system')
logging.set_verbosity_error()
log = get_pylogger(__name__)


if __name__ == "__main__":

    seed_everything(42, workers=True)

    # ------------------------
    # SETTINGS
    # ------------------------

    # set up CLI args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_type",
                           type=str,
                           default="text",
                           help="input type of the model, "
                           "only support 'id' and 'text'")
    
    # set program args
    temp_args, _ = argparser.parse_known_args()
    argparser = add_program_args(temp_args, argparser)
    
    # set model and dataset args
    temp_args, _ = argparser.parse_known_args()
    datamodule = get_datamodule(args=temp_args)
    argparser = datamodule.add_datamodule_specific_args(parent_parser=argparser)

    # parse args
    args, _ = argparser.parse_known_args()

    # set up datamodule
    datamodule_config = datamodule.build_datamodule_config(args=args)
    dm = datamodule(datamodule_config)

    # prepare data
    num_items = dm.prepare_data()
 