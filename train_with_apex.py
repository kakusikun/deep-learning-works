import argparse

from tools.logger import setup_logger
from src.factory.config_factory import _C as cfg
from src.factory.config_factory import build_output, show_products
from tools.utils import deploy_macro, print_config

from src.factory.loader_factory import LoaderFactory
from src.factory.graph_factory import GraphFactory
from src.solver.solver import Solver
from src.factory.engine_factory import EngineFactory

from apex.fp16_utils import *
from apex import amp, optimizers

def main():
    parser = argparse.ArgumentParser(description="PyTorch Deep Learning")
    parser.add_argument("--config", default="", help="path to config file", type=str)
    parser.add_argument('--products', action='store_true',
                        help='list available products in all factories')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.products:
        show_products()

    if args.config != "":
        cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)    
    build_output(cfg, args.config)
    logger = setup_logger(cfg.OUTPUT_DIR)
    deploy_macro(cfg)

    loader = LoaderFactory.produce(cfg)
    graph = GraphFactory.produce(cfg)
    solver = Solver(cfg, graph.model.named_parameters())

    graph.model = graph.model.cuda()

    p = next(iter(graph.model.parameters()))
    print(p.is_cuda)

    graph.model, solver.opt = amp.initialize(graph.model, solver.opt,
                                  opt_level='O3',
                                  keep_batchnorm_fp32=True
                                  )

    graph.use_multigpu()                                 
    p = next(iter(graph.model.parameters()))
    print(p.dtype)

    batch = next(iter(loader['val']))
    for key in batch:
        batch[key] = batch[key].cuda()
    
    solver.zero_grad()
    outputs = graph.model(batch['inp'])

    loss, losses = graph.loss_head(outputs, batch)
    with amp.scale_loss(loss, solver.opt) as scaled_loss:
        scaled_loss.backward()
    solver.step()

if __name__ == "__main__":
    main()