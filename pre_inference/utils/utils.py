from datamodules import SeqDataModule, PreInferSeqDataModule
from utils.cli_parse import parse_boolean

def get_datamodule(args):
    # add new datamodule if needed
    input_type = args.input_type
    if input_type == 'id':
        data_module = SeqDataModule
    elif input_type == 'text':
        if args.pre_inference:
            data_module = PreInferSeqDataModule
        else:
            data_module = SeqDataModule
    else:
        raise NotImplementedError
    return data_module


def add_program_args(args, parent_parser):
    parser = parent_parser.add_argument_group("program")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--early_stop_patience",
                        type=int,
                        default=10,
                        help="early stop patience")
    parser.add_argument(
        "--strategy",
        type=str,
        default="none",
        help="specify deepspeed stage, support stage_2, stage_2_offload"
        "stage_3, stage_3_offload, none")

    if args.input_type == 'text':
        parser.add_argument("--plm_name",
                            type=str,
                            default="facebook/opt-125m")
        parser.add_argument("--use_prompt",
                            type=parse_boolean,
                            default=False,
                            help="whether to use prompt")
        parser.add_argument("--pre_inference",
                            type=parse_boolean,
                            default=True,
                            help="whether to pre-inference")

    return parent_parser

def read_distributed_strategy(args):
    if args.strategy == "none":
        strategy = "ddp" if len(args.devices) > 1 else None
    elif args.strategy == "ddp_find_unused_parameters_false":
        strategy = "ddp_find_unused_parameters_false"
    elif args.strategy == "stage_2":
        strategy = "deepspeed_stage_2"
    elif args.strategy == "stage_2_offload":
        strategy = "deepspeed_stage_2_offload"
    elif args.strategy == "stage_3":
        strategy = "deepspeed_stage_3"
    elif args.strategy == "stage_3_offload":
        strategy = "deepspeed_stage_3_offload"
    else:
        raise ValueError("Unsupport strategy: {}".format(args.strategy))
    return strategy
