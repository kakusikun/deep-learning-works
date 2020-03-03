from src.engine import *

class ClassificationEngine(BaseEngine):
    def __init__(self, cfg, graph, loader, solvers, visualizer):
        super(ClassificationEngine, self).__init__(cfg, graph, loader, solvers, visualizer)
        
    def _train_once(self):
        accus = []   
        for i, batch in enumerate(self.tdata):
            self._train_iter_start()
            for key in batch:
                batch[key] = batch[key].cuda()
            images = batch['inp']            
            outputs = self.graph.model(images)
            self.loss, self.losses = self.graph.loss_head(outputs, batch)
            self.loss.backward()
            self._train_iter_end()     
            self.loss = self.tensor_to_scalar(self.loss)
            self.losses = self.tensor_to_scalar(self.losses)
            accus.append((outputs.max(1)[1] == batch['target']).float().mean())
            if i % 10 == 0:
                logger.info(f"Epoch [{self.epoch:03}/{self.max_epoch:03}]   Step [{i:04}/{self.cfg.SOLVER.ITERATIONS_PER_EPOCH:04}]   loss {self.loss:3.3f}")

        self.train_accu = self.tensor_to_scalar(torch.stack(accus).mean())
            

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        accus = []        
        with torch.no_grad():
            self._eval_epoch_start()
            for batch in self.vdata: 
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
        
