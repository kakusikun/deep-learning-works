from src.engine import *
from tqdm import tqdm
from tools.eval_reid_metrics import evaluate, eval_recall

# recover = T.Compose([T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229,1/0.224,1/0.225])])

class IAPReIDEngine(BaseEngine):
    def __init__(self, cfg, graph, loader, solvers, visualizer):
        super(IAPReIDEngine, self).__init__(cfg, graph, loader, solvers, visualizer)

    def _train_epoch_start(self):
        self.epoch += 1
        logger.info(f"Epoch {self.epoch} start")
        self.graph.model.train() 
        if len(self.graph.sub_models) > 0:
            for sub_model in self.graph.sub_models:
                self.graph.sub_models[sub_model].train()
        
        if self.cfg.SOLVER.MODEL_FREEZE_PEROID > 0:
            if self.epoch - 1 < self.cfg.SOLVER.MODEL_FREEZE_PEROID:
                for n, m in self.graph.model.named_modules():
                    if 'iap' in n:
                        m.train()
                        for p in m.parameters():
                            p.requires_grad = True
                        logger.info(f"{n} is trainable")
                    else:
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad = False
            elif self.epoch - 1 == self.cfg.SOLVER.MODEL_FREEZE_PEROID:
                logger.info("Model is unfreezed")
                for p in self.graph.model.parameters():
                    p.requires_grad = True
                
    def _train_once(self):
        accus = [] 
        for batch in tqdm(self.tdata, desc=f"TRAIN[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"):
            self._train_iter_start()
            if self.use_gpu:
                for key in batch:
                    if self.cfg.DISTRIBUTED:
                        batch[key] = batch[key].to(self.device, non_blocking=True)
                    else:
                        batch[key] = batch[key].cuda()
            output = self.graph.run(batch['inp']) 
            loss, losses = self.graph.loss_head(output, batch)
            self.loss, self.losses = loss, losses

            accu = (output.max(1)[1] == batch['pid']).float().mean()
            accus.append(accu)        
            self.train_accu = self.tensor_to_scalar(accu)
            self._train_iter_end()

    def _evaluate(self, eval=False):
        logger.info("Epoch {} evaluation start".format(self.epoch))
        title = "EVALUATE" if eval else f"TEST[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]"
        accus = []        
        with torch.no_grad():
            self._eval_epoch_start()
            qf, q_pids, q_camids = [], [], []
            for batch in tqdm(self.qdata, desc=title): 
                imgs, pids, camids = batch['inp'], batch['pid'], batch['camid']
                if self.cfg.DISTRIBUTED:
                    features = self.graph.run(imgs.to(self.device, non_blocking=True) if self.use_gpu else imgs)
                else:
                    features = self.graph.run(imgs.cuda() if self.use_gpu else imgs)
                qf.append(features.cpu())
                q_pids.extend(pids)
                q_camids.extend(camids)

            qf = torch.cat(qf, 0)
            q_pids = np.asarray(q_pids)
            q_camids = np.asarray(q_camids)
            logger.info("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

            gf, g_pids, g_camids = [], [], []
            for batch in tqdm(self.gdata, desc=title): 
                imgs, pids, camids = batch['inp'], batch['pid'], batch['camid']
                if self.cfg.DISTRIBUTED:
                    features = self.graph.run(imgs.to(self.device, non_blocking=True) if self.use_gpu else imgs)
                else:
                    features = self.graph.run(imgs.cuda() if self.use_gpu else imgs)
                gf.append(features.cpu())
                g_pids.extend(pids)
                g_camids.extend(camids)

            gf = torch.cat(gf, 0)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)
            logger.info("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        distmat =  1 - F.linear(qf, gf)
        distmat = distmat.numpy()

        logger.info("Computing CMC and mAP")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        logger.info("Results ----------")
        logger.info("mAP: {:.1%}".format(mAP))
        logger.info("CMC curve")
        for r in [1, 5, 10, 20]:
            logger.info("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        logger.info("------------------")

        logger.info("Computing Recall")
        rs, confs, gts, fg = eval_recall(distmat, q_pids, g_pids, q_camids, g_camids)

        logger.info("Results ------------: {:>4} / {:>4} / {:>4}".format("Q0.5", "Q0.75", "Q0.95"))
        logger.info("Number of candidates: {:.2f} / {} / {}".format(np.quantile(rs, q = 0.5), np.quantile(rs, q = 0.75), np.quantile(rs, q = 0.95)))
        logger.info("          Confidence: {:.2f} / {:.2f} / {:.2f}".format(np.quantile(confs, q = 0.5), np.quantile(confs, q = 0.75), np.quantile(confs, q = 0.95)))
        logger.info("    Number of target: {:.2f} / {} / {}".format(np.quantile(gts, q = 0.5), np.quantile(gts, q = 0.75), np.quantile(gts, q = 0.95)))  
        logger.info("------------------")
        
        if eval:
            np.save(self.cfg.OUTPUT_DIR+"/rs.npy", rs)
            np.save(self.cfg.OUTPUT_DIR+"/confs.npy", confs)
            np.save(self.cfg.OUTPUT_DIR+"/gts.npy", gts)
            np.save(self.cfg.OUTPUT_DIR+"/filtered_gallery.npy", fg)
        else:
            self.accu = cmc[0]
            self._eval_epoch_end()

        del qf, gf, distmat
        
    def Evaluate(self):
        self._evaluate(eval=True)
        logger.info(self.accu)
