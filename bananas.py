import os
import copy
import random
import pickle
import numpy as np
import tensorflow as tf
from cell_module.ops import OPS
from models.model import Model
from utils.losses import *
from utils.metrics import *
from torch.utils.data import DataLoader
from utils.acquisition_functions import acq_fn
from utils.meta_neural_net import MetaNeuralnet
from utils.dataset import vessel_dataset

class BANANAS():

    def __init__(self,
                metann_params = None,
                num_init=10, 
                k=10, 
                loss='val_loss',
                total_queries=150, 
                num_ensemble=5, 
                acq_opt_type='mutation',
                num_arches_to_mutate=1,
                max_mutation_rate=1,
                explore_type='its',
                predictor='bananas',
                predictor_encoding='trunc_path',
                mutate_encoding='adj',
                random_encoding='adj',
                verbose=1):

        self.metann_params = metann_params
        self.num_init = num_init
        self.k = k
        self.loss = loss
        self.total_queries = total_queries
        self.num_ensemble = num_ensemble
        self.acq_opt_type = acq_opt_type
        self.num_arches_to_mutate = num_arches_to_mutate
        self.max_mutation_rate = max_mutation_rate
        self.explore_type = explore_type
        self.predictor = predictor
        self.predictor_encoding = predictor_encoding
        self.nbr_ops = len(OPS) - 1
        self.cutoff = sum(self.nbr_ops**i for i in range(4))
        self.mutate_encoding = mutate_encoding
        self.random_encoding = random_encoding
        self.solNo = 0
        self.totalTrainedModel = 0

        self.NUM_EDGES = 9
        self.NUM_VERTICES = 7
        self.DIMENSIONS = 28
        self.MAX_NUM_CELL = 5
        self.OP_SPOTS = self.NUM_VERTICES - 2
        self.CELLS = [i for i in range(2, self.MAX_NUM_CELL + 1)] # 2, 3, 4, 5
        self.FILTERS = [2**i for i in range(3, 6)] # 8, 16, 32
        self.OPS = [idx for idx, op in enumerate(list(OPS.keys())[:-1])]

    def seed_torch(self, seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_param_value(self, value, step_size):
        ranges = np.arange(start=0, stop=1, step=1/step_size)
        return np.where((value < ranges) == False)[0][-1]

    def vector_to_config(self, vector):
        '''Converts numpy array to discrete values'''

        try:
            config = np.zeros(self.DIMENSIONS, dtype='uint8')
            
            max_edges = int(((self.NUM_VERTICES) * (self.NUM_VERTICES - 1)) / 2)
            # Edges
            for idx in range(max_edges):
                config[idx] = self.get_param_value(vector[idx], 2)

            # Vertices - Ops
            for idx in range(max_edges, max_edges + self.NUM_VERTICES - 2):
                config[idx] = self.get_param_value(vector[idx], len(OPS) - 1)

            # Number of Cells
            idx = max_edges + self.NUM_VERTICES - 2
            config[idx] = self.get_param_value(vector[idx], len(self.CELLS))
            
            # Number of Filters 
            config[idx + 1] = self.get_param_value(vector[idx + 1], len(self.FILTERS))
        except Exception as e:
            print("HATA...", vector, e)

        return config

    def init_rnd_nbr_generators(self):
        # Random Number Generators
        self.sample_pop_rnd = np.random.RandomState(seed)
        self.init_pop_rnd = np.random.RandomState(seed)
    
    def writePickle(self, data, name):
        # Write History
        with open(f"results/{path}/model_{name}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)

    def f_objective(self, model):
        
        fitness, cost, log = model.evaluate(train_loader, val_loader, loss_fn, metric_fn, device)
        if fitness != -1:
            self.totalTrainedModel += 1
            self.writePickle(model, model.solNo)
            with open(f"results/{path}/model_{model.solNo}.txt", "w") as f:
                f.write(log)
        return fitness, cost



    def mutate_arch(self, 
                    arch, 
                    mutation_rate=1.0, 
                    mutate_encoding='adj',
                    cutoff=0):

        return self.mutate(arch,
                           mutation_rate=mutation_rate,
                           mutate_encoding=mutate_encoding,
                           cutoff=cutoff)
    

    def adj_mutate(self,
               matrix,
               ops,
               cont,
               cells,
               filters,
               mutation_rate=1.0,
               patience=5000):
    
        p = 0
        while p < patience:
            p += 1
            new_matrix = copy.deepcopy(matrix)
            new_ops = copy.deepcopy(ops)
            new_ops.insert(0, 'input')
            new_ops.append('output')

            if not cont:
                # flip each edge w.p. so expected flips is 1. same for ops
                edge_mutation_prob = mutation_rate / (self.NUM_VERTICES * (self.NUM_VERTICES - 1) / 2)
                for src in range(0, self.NUM_VERTICES - 1):
                    for dst in range(src + 1, self.NUM_VERTICES):
                        if np.random.rand() < edge_mutation_prob:
                            new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutation_rate / self.OP_SPOTS
            for ind in range(1, self.OP_SPOTS + 1):
                if np.random.rand() < op_mutation_prob:
                    available = [o for o in self.OPS if o != new_ops[ind]]
                    new_ops[ind] = np.random.choice(available)

            new_nbr_cell = cells
            if np.random.rand() < 0.5:
                new_nbr_cell = np.random.choice(list(set(self.CELLS) - {cells}))
            
            new_nbr_filter = filters
            if np.random.rand() < 0.5:
                new_nbr_filter = np.random.choice(list(set(self.FILTERS) - {filters}))

            new_spec = Model(matrix=new_matrix, ops=new_ops[1:-1], nbr_cell = new_nbr_cell, nbr_filters = new_nbr_filter, compile=False)
            if new_spec.isFeasible:
                return new_spec
        return self.adj_mutate(matrix, ops, mutation_rate=mutation_rate+1)

    def mutate(self,
               arch, 
               mutation_rate=1.0,
               mutate_encoding='adj',
               cutoff=None,
               comparisons=2500,
               patience=5000,
               prob_wt=False):

        if mutate_encoding in ['adj', 'cont_adj']:
            cont = ('cont' in mutate_encoding)
            return self.adj_mutate(matrix=arch[1],
                              ops=arch[2],
                              cont=cont,
                              cells=arch[3],
                              filters=arch[4])

    def get_candidates(self,
                       data, 
                       num=100,
                       mutate_encoding='adj',
                       patience_factor=5, 
                       num_arches_to_mutate=1,
                       max_mutation_rate=1):

        candidates = []
        # set up hash map
        dic = {}
        for d in data:
            h = d.get_hash()
            dic[h] = 1

        # mutate architectures with the lowest loss
        best_arches = [(arch.solNo, arch.org_matrix, arch.org_ops, arch.nbr_cell, arch.nbr_filters) for arch in sorted(data, key=lambda i:i.fitness, reverse=True)[:num_arches_to_mutate * patience_factor]]

        # stop when candidates is size num
        # use patience_factor instead of a while loop to avoid long or infinite runtime

        for arch in best_arches:
            if len(candidates) >= num:
                break
            for i in range(int(num / num_arches_to_mutate / max_mutation_rate)):
                for rate in range(1, max_mutation_rate + 1):
                    mutated = self.mutate_arch(arch, 
                                                mutation_rate=rate, 
                                                mutate_encoding=mutate_encoding)
                    mutated.encoding = mutated.encode(predictor_encoding='trunc_path', cutoff=self.cutoff)
                    h = mutated.get_hash()

                    if h not in dic:
                        dic[h] = 1    
                        candidates.append(mutated)

        return candidates
        
    def generate_random_dataset(self):

        dic = {}
        data = []
        counter = 0
        
        while counter < self.num_init:
            chromosome = self.init_pop_rnd.uniform(low=0.0, high=1.0, size=self.DIMENSIONS)
            config = self.vector_to_config(chromosome)
            model = Model(chromosome, config, self.CELLS[config[-2]], self.FILTERS[config[-1]])       

            h = model.get_hash()
            if h != () and h not in dic:
                model.encoding = model.encode(predictor_encoding='trunc_path', cutoff=self.cutoff)
                dic[h] = 1
                counter += 1
                model.solNo = self.solNo
                self.solNo += 1
                data.append(model)
        
        return data

    def eval_dataset(self, data):
        for model in data:
            model.fitness, _ = self.f_objective(model)

    def start_bananas(self, num_ensemble = 5, predictor = 'bananas', metann_params = None):
        self.init_rnd_nbr_generators()
        # Generate num_init random architecture
        data = self.generate_random_dataset()
        self.eval_dataset(data)

        query = self.num_init

        while query <= 155:

            xtrain = np.array([d.encoding for d in data])
            ytrain = np.array([(1 - d.fitness) * 100 for d in data])

            candidates = self.get_candidates(data)

            xcandidates = np.array([c.encoding for c in candidates])
            candidate_predictions = []

            # train an ensemble of neural networks
            train_error = 0
            ensemble = []

            for e in range(num_ensemble):

                if predictor == 'bananas':
                    meta_neuralnet = MetaNeuralnet()
                    net_params = metann_params['ensemble_params'][e]

                    train_error += meta_neuralnet.fit(xtrain, ytrain, **net_params)

                    # predict the validation loss of the candidate architectures
                    candidate_predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))
                    tf.compat.v1.reset_default_graph()

            train_error /= num_ensemble
            print('query {}, Neural predictor train error: {}'.format(query, train_error))

            # compute the acquisition function for all the candidate architectures
            candidate_indices = acq_fn(candidate_predictions, ytrain=ytrain, explore_type=self.explore_type)

            counter = 0
            # add the k arches with the minimum acquisition function values
            for i in candidate_indices:
                
                candidates[i].solNo = self.solNo
                candidates[i].compile_model()
                fitness, _ = self.f_objective(candidates[i])
                if candidates[i].isFeasible and fitness != -1:
                    data.append(candidates[i])
                    query += 1
                    counter += 1
                    self.solNo += 1
                
                if counter >= self.k:
                    break

            del candidates
            tf.keras.backend.clear_session()

            top_5_loss = sorted([(1 - d.fitness) * 100 for d in data])[:min(5, len(data))]
            print('{}, query {}, top 5 losses {}'.format(predictor, query, top_5_loss))


    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    device = torch.device('cuda')
    
    data_path = "DataSets/DRIVE"
    batch_size = 128
    seed = 42

    # Main
    metanet_params = {'loss':'mae', 'num_layers':10, 'layer_width':20,'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
    params = {'ensemble_params':[metanet_params for _ in range(5)]}
	
    train_dataset = vessel_dataset(data_path, mode="training", split=0.9, de_train=True)
    val_dataset = vessel_dataset(data_path, mode="training", split=0.9, is_val=True, de_train=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    
    print("Number of training image:", train_dataset.__len__())
    
    path = f"bananas_{seed}"
    if not os.path.exists(f"results/{path}/"):
        os.makedirs(f"results/{path}/")
    
    loss_fn = DiceLoss()
    metric_fn = DiceCoef()

    bananas = BANANAS()
    bananas.seed_torch(seed)
    bananas.start_bananas(metann_params=params)
