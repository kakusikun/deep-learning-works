import os
import random
import time
import json
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import tensor_to_scalar

class Evolution:
    def __init__(self, 
        cfg,
        graph, 
        pool_target_size=20, 
        children_size=10, 
        parent_size=5, 
        mutate_ratio=0.1,  
        flops_cuts=7,
        children_pick_interval=3, 
        logger=None):
        self.cfg = cfg
        self.graph = graph
        self.pool_target_size = pool_target_size
        self.children_size = children_size
        self.parent_size = parent_size
        self.mutate_ratio = mutate_ratio
        self.flops_cuts = flops_cuts
        self.parents = []
        self.children = []       

        self.lookup_table = self.graph.lookup_table
        max_flops, min_flops, max_params, min_params = self.set_flops_params_bound()
        # [top to bottom then bottom to top] 
        self.flops_interval = (max_flops - min_flops) / flops_cuts
        self.flops_ranges = [max(max_flops - i * self.flops_interval, 0) for i in range(flops_cuts)] + \
                            [max(max_flops - i * self.flops_interval, 0) for i in range(flops_cuts)][::-1]

        # Use worse children of the good parents
        # If the children are too outstanding, the distribution coverage ratio will be low
        # [0, 3, 6, 9, 9, 6, 3, 0] => [6, 6, 6, 9, 9, 6, 6, 6]
        children_pick_ids = list(range(0, children_size, children_pick_interval)) + \
                                 list(reversed(range(0, children_size, children_pick_interval)))
        self.children_pick_ids = [6 if idx == 0 or idx == 3 else idx for idx in children_pick_ids]

        self.sample_counts = cfg.SOLVER.ITERATIONS_PER_EPOCH // len(self.flops_ranges) // len(self.children_pick_ids)

        self.param_interval = (max_params - min_params) / (len(self.children_pick_ids) - 1)
        # [top to bottom] 
        self.param_range = [max_params - i * self.param_interval for i in range(len(self.children_pick_ids))]

        self.cur_step = 0
        
        p = next(iter(self.graph.model.parameters()))
        if p.is_cuda:
            self.use_gpu = True

        self.bad_generations = []

    def evolve(self, epoch_after_search, pick_id, find_max_param, max_flops, max_params, min_params, logger=None):
        '''
        Returns:
            selected_child(dict):
                a candidate that 
        '''
        g = f"{find_max_param}, {max_flops:.2f}, {max_params:.2f}, {min_params:.2f}"
        while g in self.bad_generations:
            pick_id, find_max_param, max_flops, max_params, min_params = self.forced_evolution() 
            g = f"{find_max_param}, {max_flops:.2f}, {max_params:.2f}, {min_params:.2f}"
        # Prepare random parents for the initial evolution
        block_candidates = self.graph.generate_block_candidates(epoch_after_search)
        channel_candidates = self.graph.generate_channel_candidates(epoch_after_search)
        while len(self.parents) < self.parent_size:
            block_choices = self.graph.random_block_choices(epoch_after_search)
            channel_choices = self.graph.random_channel_choices(epoch_after_search)
            flops, param = get_flop_params(block_choices, channel_choices, self.lookup_table)
            candidate = dict()
            candidate['block_choices'] = block_choices
            candidate['channel_choices'] = channel_choices
            candidate['flops'] = flops
            candidate['param'] = param
            self.parents.append(candidate)

        # Breed children
        duration = 0.0
        while len(self.children) < self.children_size:
            start = time.time()
            candidate = dict()
            # randomly select parents from current pool
            mother = random.choice(self.parents)
            father = random.choice(self.parents)

            # make sure mother and father are different
            while father is mother:
                mother = random.choice(self.parents)

            # breed block choice
            block_choices = [0] * len(father['block_choices'])
            for i in range(len(block_choices)):
                block_choices[i] = random.choice([mother['block_choices'][i], father['block_choices'][i]])
                # Mutation: randomly mutate some of the children.
                if random.random() < self.mutate_ratio:
                    block_choices[i] = random.choice(block_candidates[i])

            # breed channel choice
            channel_choices = [0] * len(father['channel_choices'])
            for i in range(len(channel_choices)):
                channel_choices[i] = random.choice([mother['channel_choices'][i], father['channel_choices'][i]])
                # Mutation: randomly mutate some of the children after all channel is warming up.
                if random.random() < self.mutate_ratio and epoch_after_search > 0:
                    channel_choices[i] = random.choice(channel_candidates[i])

            flops, param = get_flop_params(block_choices, channel_choices, self.lookup_table)

            if epoch_after_search > 0:
                # if flops > max_flop or model_size > upper_params:
                if flops < (max_flops-self.flops_interval) or flops > max_flops \
                        or param < min_params or param > max_params:
                    duration += time.time() - start
                    if duration > (self.cfg.SPOS.DURATION - 1): # cost too much time in evolution
                        if logger:
                            logger.info("Give up this generation for wasting too much time")
                            logger.info(g)
                        self.bad_generations.append(g)
                        self._record_bad_generation()   
                        pick_id, find_max_param, max_flops, max_params, min_params = self.forced_evolution()                     
                        g = f"{find_max_param}, {max_flops:.2f}, {max_params:.2f}, {min_params:.2f}"
                        while g in self.bad_generations:
                            pick_id, find_max_param, max_flops, max_params, min_params = self.forced_evolution() 
                            g = f"{find_max_param}, {max_flops:.2f}, {max_params:.2f}, {min_params:.2f}"
                        duration = 0.0
                    start = time.time()    
                    print(f"\r Evolving {int(duration)}s", end = '')
                    continue

            candidate['block_choices'] = block_choices
            candidate['channel_choices'] = channel_choices
            candidate['flops'] = flops
            candidate['param'] = param
            self.children.append(candidate)
        # Set target and select
        self.children.sort(key=lambda cand: cand['param'], reverse=find_max_param)
        selected_child = self.children[pick_id]

        # Update step for the strolling evolution
        self.cur_step += 1

        # prepare for next evolve
        self.parents = self.children[:self.parent_size]
        self.children = []

        return selected_child

    def forced_evolution(self):
        self.cur_step += 1
        max_flops, pick_id, range_id, find_max_param = self.get_cur_evolve_state()
        if find_max_param:
            max_params=self.param_range[range_id]
            min_params=self.param_range[-1]
        else:
            max_params=self.param_range[0]
            min_params=self.param_range[range_id]
        return pick_id, find_max_param, max_flops, max_params, min_params

    def get_cur_evolve_state(self):
        '''
        walk(cur_step) on the map(flop x param) from large param to small given flop from which large to small then
        walk back.
        '''
        self.cur_step = self.cur_step % (self.sample_counts * len(self.children_pick_ids) * len(self.flops_ranges))
        i = self.cur_step // (len(self.children_pick_ids) * self.sample_counts)
        j = self.cur_step % (len(self.children_pick_ids) * self.sample_counts) // self.sample_counts
        range_id = j if i % 2 == 0 else len(self.children_pick_ids) - 1 - j
        find_max_param = False
        if (i % 2 == 0 and j < len(self.children_pick_ids) // 2) or \
                (not i % 2 == 0 and j >= len(self.children_pick_ids) // 2):
            find_max_param = True
        return self.flops_ranges[i], self.children_pick_ids[j], range_id, find_max_param

    def maintain(self, epoch_after_search, pool, lock, finished_flag, logger=None):
        self._read_bad_generation()
        logger.info("Evolution Starts")
        while not finished_flag.value:
            if len(pool) < self.pool_target_size:
                max_flops, pick_id, range_id, find_max_param = self.get_cur_evolve_state()
                if find_max_param:
                    info = f"[Evolution] Find max params   Max Flops [{max_flops:.2f}]   Child Pick ID [{pick_id}]   Upper model size [{self.param_range[range_id]:.2f}]   Bottom model size [{self.param_range[-1]:.2f}]" 
                    if logger and self.cur_step % self.sample_counts == 0 and epoch_after_search > 0:
                        logger.info('-' * 40 + '\n' + info)
                    candidate = self.evolve(
                        epoch_after_search,
                        pick_id, 
                        find_max_param, 
                        max_flops,
                        max_params=self.param_range[range_id],
                        min_params=self.param_range[-1],
                        logger=logger
                    )
                else:
                    info = f"[Evolution] Find min params   Max Flops [{max_flops:.2f}]   Child Pick ID [{pick_id}]   Upper model size [{self.param_range[range_id]:.2f}]   Bottom model size [{self.param_range[-1]:.2f}]" 
                    if logger and self.cur_step % self.sample_counts == 0 and epoch_after_search > 0:
                        logger.info('-' * 40 + '\n' + info)
                    candidate = self.evolve(
                        epoch_after_search,
                        pick_id, 
                        find_max_param, 
                        max_flops,
                        max_params=self.param_range[0],
                        min_params=self.param_range[range_id],
                        logger=logger
                    )
                with lock:
                    pool.append(candidate)
        logger.info("[Evolution] Ends")

    def set_flops_params_bound(self):
        block_choices = [2] * sum(self.graph.stage_repeats)
        channel_choices = [7] * sum(self.graph.stage_repeats)
        max_flops, max_params = get_flop_params(block_choices, channel_choices, self.lookup_table)
        block_choices = [0] * sum(self.graph.stage_repeats)
        cum_repeats = 0
        for repeats in self.graph.stage_repeats:
            block_choices[cum_repeats] = 1
            cum_repeats += repeats
        channel_choices = [3] * sum(self.graph.stage_repeats)     
        min_flops, min_params = get_flop_params(block_choices, channel_choices, self.lookup_table)
        return max_flops, min_flops, max_params, min_params

    def _record_bad_generation(self):
        root = os.path.join(os.getcwd(), "external")
        if not os.path.exists(root):
            os.makedirs(root)
        path = os.path.join(root, 'spos_historical_bad_generations.txt')
        self._read_bad_generation()
        with open(path, 'w') as f:
            for g in self.bad_generations:
                msg = g + "\n"
                f.write(msg)

    def _read_bad_generation(self):
        root = os.getcwd()
        path = os.path.join(root, 'external/spos_historical_bad_generations.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f.readlines():
                    g = line.strip()
                    self.bad_generations.append(g)

class SearchEvolution:
    def __init__(self, 
        cfg, 
        graph, 
        vdata,
        bndata,
        population_size=500, 
        retain_length=100, 
        random_select=0.1, 
        mutate_chance=0.1,
        bn_recalc_imgs=20000,
        logger=None):
        self.cfg = cfg
        self.graph = graph
        self.vdata = vdata
        self.bndata = bndata
        self.population_size = population_size
        self.retain_length = retain_length
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.bn_recalc_imgs = bn_recalc_imgs
        self.logger = logger
        self.history = defaultdict(list)

        self.lookup_table = self.graph.lookup_table
        self.graph.to_gpus()
        self.block_candidates = self.graph.generate_block_candidates()
        self.channel_candidates = self.graph.generate_channel_candidates()

    def build_population(self):
        population = []
        start = time.time()
        while len(population) < self.population_size:
            block_choices = self.graph.random_block_choices()
            channel_choices = self.graph.random_channel_choices()
            flops, param = get_flop_params(block_choices, channel_choices, self.lookup_table)
            instance = {}
            instance['block'] = block_choices
            instance['channel'] = channel_choices
            instance['flops'] = flops
            instance['param'] = param
            population.append(instance)
            print("\r Building Population" + f"{int(time.time()-start)}s", end='')
        if self.logger:
            self.logger.info("Population Built")
        return population

    def _get_choice_accuracy(self, block_choices, channel_choices):
        self.graph.model.train()
        recalc_bn(self.graph, block_choices, channel_choices, self.bndata, True, self.bn_recalc_imgs)
        self.graph.model.eval()
        accus = []
        start = time.time()
        msg = "Testing"
        for batch in self.vdata:
            iter_start = time.time()
            with torch.no_grad():
                for key in batch:
                    batch[key] = batch[key].cuda()
                outputs = self.graph.model(batch['inp'], block_choices, channel_choices)
            accus.append((outputs.max(1)[1] == batch['target']).float().mean())
            print("\r  ------------------ " + f"{msg:<20} [{time.time()-start:.2f}]s    [{time.time()-iter_start:.2f}]s/iter                      ", end='')
        accu = tensor_to_scalar(torch.stack(accus).mean())
        print(f"  ------------------ {accu:.4f}")
        return accu

    def born(self, father, mother):
        children = []
        for _ in range(2):
            child = {}
            child['block'] = self.crossover_mutate(father['block'], mother['block'], self.block_candidates, self.mutate_chance)
            child['channel'] = self.crossover_mutate(father['channel'], mother['channel'], self.channel_candidates, self.mutate_chance)
            children.append(child)
        return children

    def evolve(self, population, leader_board, search_iter):
        start = time.time()
        for i, instance in enumerate(population):
            if 'error' not in instance:
                if self.logger and i % 50 == 0:
                    self.logger.info(f"Growing    Search [{search_iter:03}]    Step [{i:03}]    Duration [{(time.time()-start)/60:.2f}]s")

                acc = self._get_choice_accuracy(instance['block'], instance['channel'])
                instance['error'] = 1 - acc
                temp = (
                    deepcopy(instance['error']), 
                    deepcopy(instance['block']),
                    deepcopy(instance['channel']),
                    deepcopy(instance['flops']),
                    deepcopy(instance['param']),
                )
                leader_board.push(temp)
                self.history[f"{acc:.3f}"].append(instance)

        population.sort(key=lambda x: x['error'])
        parents = population[:self.retain_length]
        for instance in population[self.retain_length:]:
            if self.random_select > random.random():
                parents.append(instance)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children_to_be_grown = []
        
        # Add children, which are bred from two remaining networks.
        while len(children_to_be_grown) < desired_length:
            # Get a random mom and dad.
            father = random.randint(0, parents_length-1)
            mother = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
            if father != mother:
                father = parents[father]
                mother = parents[mother]

                # Breed them.
                children = self.born(father, mother)

                # Add the children one at a time.
                for child in children:
                    # Don't grow larger than desired length.
                    if len(children_to_be_grown) >= desired_length:
                        break
                    flops, param = get_flop_params(child['block'], child['channel'], self.lookup_table)
                    child['flops'] = flops
                    child['param'] = param
                    children_to_be_grown.append(child)
        if self.logger:
            self.logger.info(f"Evolved    Search [{search_iter:03}]")
        parents.extend(children_to_be_grown)
        self._record_search_history(search_iter)
        return parents

    def evolve_paper(self, population, leader_board, search_iter):
        for instance in population:
            if 'error' not in instance:
                acc = self._get_choice_accuracy(instance['block'], instance['channel'])
                instance['error'] = 1 - acc
                temp = (
                    1 - deepcopy(instance['error']), 
                    deepcopy(instance['block']),
                    deepcopy(instance['channel']),
                    deepcopy(instance['flops']),
                    deepcopy(instance['param']),
                )
                leader_board.push(temp)        
                self.history[f"{acc:.3f}"].append(instance)
        population = leader_board.topk()
        # Now find out how many spots we have left to fill.
        start = time.time()
        p_crossover = self.mass_crossover(population[:(len(population) // 2)], self.population_size // 2)
        p_mutation = self.mass_mutation(population[(len(population) // 2):], self.population_size - self.population_size // 2)
        population = p_crossover + p_mutation
        print("\r Evolving" + f"{int(time.time()-start)}s", end='')
        if self.logger:
            self.logger.info(f"[{search_iter:03}] Population Evolved")
        self._record_search_history(search_iter)
        return population

    def mass_crossover(self, population, size):
        children = []
        while len(children) < size:
            father = random.randint(0, len(population)-1)
            mother = random.randint(0, len(population)-1)
            # Assuming they aren't the same network...
            if father != mother:
                father = population[father]
                mother = population[mother]
                child = {}
                child['block'] = self.crossover_mutate(father['block'], mother['block'], self.graph.block_candidates, -1)
                child['channel'] = self.crossover_mutate(father['channel'], mother['channel'], self.graph.channel_candidates, -1)
                flops, param = get_flop_params(child['block'], child['channel'], self.lookup_table)
                child['flops'] = flops
                child['param'] = param
            children.append(child)
        return children

    def mass_mutation(self, population, size):
        alien = []
        while len(alien) < size:
            instance = population[random.randint(0, len(population)-1)]
            instance['block'] = self.crossover_mutate(instance['block'], instance['block'], self.graph.block_candidates, self.mutate_chance)
            instance['channel'] = self.crossover_mutate(instance['channel'], instance['channel'], self.graph.channel_candidates, self.mutate_chance)
            flops, param = get_flop_params(instance['block'], instance['channel'], self.lookup_table)
            instance['flops'] = flops
            instance['param'] = param
            alien.append(instance)
        return alien

    def _record_search_history(self, search_iter):
        root = os.path.join(os.getcwd(), "external")
        if not os.path.exists(root):
            os.makedirs(root)
        path = os.path.join(self.cfg.OUTPUT_DIR, f"spos_search_history_{search_iter:03}.json")
        with open(path, 'w') as f:
            json.dump(self.history, f)

    @staticmethod
    def crossover_mutate(a, b, choices, prob):
        c = [0] * len(a)
        for i in range(len(a)):
            c[i] = random.choice([b[i], a[i]])
            if prob > random.random():
                c[i] = random.choice(choices[i])
        return c

def recalc_bn(graph, block_choices, channel_choices, bndata, use_gpu, bn_recalc_imgs=20000):
    count = 0
    start = time.time()
    msg = "BN Updating"
    for batch in bndata:
        iter_start = time.time()
        if use_gpu:
            img = batch['inp'].cuda()
        graph.model(img, block_choices, channel_choices)

        count += img.size(0)        
        print("\r  ------------------ " + f"{msg:<20} [{time.time()-start:.2f}]s    [{time.time()-iter_start:.2f}]s/iter    [{count / bn_recalc_imgs * 100:.2f}]%", end='')
        if count > bn_recalc_imgs:
            break

def get_flop_params(all_block_choice, all_channel_choice, lookup_table):
    assert isinstance(all_block_choice, list) and isinstance(all_channel_choice, list)
    flops = params = 0.0
    for block_idx, (block_choice, channel_choice) in enumerate(zip(all_block_choice, all_channel_choice)):
        choice_id = f"{block_idx}-{block_choice}-{channel_choice}"
        flops += lookup_table['flops']['backbone'][choice_id]
        params += lookup_table['params']['backbone'][choice_id]
    return flops, params
    
if __name__ == "__main__":
    import torch
    from src.factory.config_factory import _C as cfg
    from src.graph.spos import SPOS

    cfg.INPUT.RESIZE = (112, 112)
    cfg.DB.NUM_CLASSES = 10
    cfg.SPOS.USE_SE = True
    cfg.SPOS.LAST_CONV_AFTER_POOLING = True
    cfg.SPOS.CHANNELS_LAYOUT = "OneShot"
    cfg.SPOS.DURATION = 5
    graph = SPOS(cfg)

    evolution = Evolution(cfg, graph)

    max_flops, pick_id, range_id, find_max_param = evolution.get_cur_evolve_state()

    if find_max_param:    
        candidate = evolution.evolve(50, pick_id, find_max_param, max_flops,
                                max_params=evolution.param_range[range_id],
                                min_params=evolution.param_range[-1])
    else:   
        candidate = evolution.evolve(50, pick_id, find_max_param, max_flops,
                                max_params=evolution.param_range[0],
                                min_params=evolution.param_range[range_id])
