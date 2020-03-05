from src.engine import *
from tqdm import tqdm

class ClassificationEngine(BaseEngine):
    def __init__(self, cfg, graph, loader, solvers, visualizer):
        super(ClassificationEngine, self).__init__(cfg, graph, loader, solvers, visualizer)
        
    def _train_once(self):
        accus = []   
        for batch in tqdm(self.tdata, desc=f"TRAIN[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"):
            self._train_iter_start()
            for key in batch:
                batch[key] = batch[key].cuda()
            images = batch['inp']            
            outputs = self.graph.model(images)
            self.loss, self.losses = self.graph.loss_head(outputs, batch)
            accus.append((outputs.max(1)[1] == batch['target']).float().mean())        
            self._train_iter_end()    

        self.train_accu = self.tensor_to_scalar(torch.stack(accus).mean())
            
    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        title = "EVALUATE" if eval else f"TEST[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"
        accus = []        
        with torch.no_grad():
            self._eval_epoch_start()
            for batch in tqdm(self.vdata, desc=title): 
                for key in batch:
                    batch[key] = batch[key].cuda()
                images = batch['inp']      
                
                outputs = self.graph.model(images)

                accus.append((outputs.max(1)[1] == batch['target']).float().mean())
          
        self.accu = self.tensor_to_scalar(torch.stack(accus).mean())    

        if not eval:
            self._eval_epoch_end()        

    def Evaluate(self):
        self._evaluate(eval=True)
        logger.info(self.accu)
        
