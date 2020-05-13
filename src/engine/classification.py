from src.engine import *
from tqdm import tqdm

class ClassificationEngine(BaseEngine):
    def __init__(self, cfg, graph, loader, solvers, visualizer):
        super(ClassificationEngine, self).__init__(cfg, graph, loader, solvers, visualizer)
        
    def _train_once(self):
        for batch in tqdm(self.tdata, desc=f"TRAIN[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"):
            self._train_iter_start()
            if self.use_gpu:
                for key in batch:
                    if self.cfg.DISTRIBUTED:
                        batch[key] = batch[key].to(self.device, non_blocking=True)
                    else:
                        batch[key] = batch[key].cuda()
            output = self.graph.run(batch['inp']) 
            self.loss, self.losses = self.graph.loss_head(output, batch)
            self.train_accu = self.tensor_to_scalar((output.max(1)[1] == batch['target']).float().mean())
            self._train_iter_end()    
            
    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        title = "EVALUATE" if eval else f"TEST[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"
        accus = []        
        with torch.no_grad():
            self._eval_epoch_start()
            for batch in tqdm(self.vdata, desc=title): 
                if self.use_gpu:
                    for key in batch:
                        if self.cfg.DISTRIBUTED:
                            batch[key] = batch[key].to(self.device, non_blocking=True)
                        else:
                            batch[key] = batch[key].cuda()
                output = self.graph.run(batch['inp']) 
                accus.append((output.max(1)[1] == batch['target']).float().mean())
          
        self.accu = self.tensor_to_scalar(torch.stack(accus).mean())    

        if not eval:
            self._eval_epoch_end()        

    def Evaluate(self):
        self._evaluate(eval=True)
        logger.info(self.accu)
        
