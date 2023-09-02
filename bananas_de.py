import os
import copy
import random
import pickle
import numpy as np
from params import *
import tensorflow as tf
from cell_module.ops import OPS
from models.model import Model
from utils.losses import *
from utils.metrics import *
from torch.utils.data import DataLoader
from utils.dataset import vessel_dataset
from utils.acquisition_functions import acq_fn
from utils.meta_neural_net import MetaNeuralnet
from utils.drive_dataset import CustomImageDataset
from sklearn.ensemble import RandomForestRegressor

class BANANAS():

    def __init__(self,
                pop_size=10, 
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
                crossover_strategy='bin'):

        self.k = k
        self.pop_size = pop_size
        
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
        self.dic = dict()
        self.solNo = 0
        self.totalTrainedModel = 0
        self.mutation_factor = 0.5
        self.crossover_prob = 0.5
        self.crossover_strategy = crossover_strategy

        self.NUM_EDGES = 9
        self.NUM_VERTICES = 7
        self.DIMENSIONS = 28
        self.MAX_NUM_CELL = 5
        self.MAX_EDGE_NBR = int(((self.NUM_VERTICES) * (self.NUM_VERTICES - 1)) / 2)
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
        self.crossover_rnd = np.random.RandomState(seed)
    
    def writePickle(self, data, name):
        # Write History
        with open(f"results/{path}/model_{name}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)

    def f_objective(self, model):

        if (model.is_compiled is None) or (not model.is_compiled):
            model.compile_model()

        fitness, cost, log = model.evaluate(train_loader, val_loader, loss_fn, metric_fn, device)
        if fitness != -1:
            self.totalTrainedModel += 1
            self.writePickle(model, model.solNo)
            with open(f"results/{path}/model_{model.solNo}.txt", "w") as f:
                f.write(log)
        return fitness, cost


    def sample_population(self, pop, size = None):
        '''Samples 'size' individuals'''

        selection = self.sample_pop_rnd.choice(np.arange(len(pop)), size, replace=False)
        return np.array(pop)[selection].tolist()

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant
    
    def rand_to_best_2(self, best, r1, r2, r3, r4, r5):
        diff_1 = r2 - r3
        diff_2 = r4 - r5
        return r1 + (self.mutation_factor * (best - r1)) + (self.mutation_factor * (diff_1)) + (self.mutation_factor * (diff_2))

    def boundary_check(self, vector):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        projection == The invalid value is truncated to the nearest limit
        random == The invalid value is repaired by computing a random number between its established limits
        reflection == The invalid value by computing the scaled difference of the exceeded bound multiplied by two minus

        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector

        vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))

        return vector

    def get_model_dict(self, model, hash):
        return  {"solNo": model.solNo, "chromosome": model.chromosome, 
                "config": model.config, "nbr_cell": model.nbr_cell,
                "nbr_filters": model.nbr_filters, "is_compiled": model.is_compiled,
                "fitness": model.fitness, "org_matrix": model.org_matrix,
                "matrix": model.matrix, "org_ops": model.org_ops,
                "ops": model.ops, "isFeasible": model.isFeasible,
                "encoding": model.encoding, "hash": hash}

    def dict_to_model(self, dict):
        model = Model(dict["chromosome"], dict['config'],
                      dict["nbr_cell"], dict["nbr_filters"],
                      False, dict["org_matrix"],
                      dict["org_ops"])
        
        model.fitness = dict["fitness"]
        model.encoding = dict["encoding"]
        model.hash = dict["hash"]
        model.solNo = dict["solNo"]

        return model

    def init_population(self):

        dic = {}
        data = []
        counter = 0
        
        while counter < self.pop_size:
            chromosome = self.init_pop_rnd.uniform(low=0.0, high=1.0, size=self.DIMENSIONS)
            config = self.vector_to_config(chromosome)
            model = Model(chromosome, config, self.CELLS[config[-2]], self.FILTERS[config[-1]], compile=False)       

            h = model.get_hash()
            if h != () and h not in dic:
                model.encoding = model.encode(predictor_encoding='trunc_path', cutoff=self.cutoff)
                dic[h] = 1
                
                model.solNo = self.solNo
                self.solNo += 1
                data.append(self.get_model_dict(model, h))

                counter += 1
                del model
        
        return data

    def init_eval_pop(self):

        print("Start Initialization...")
        self.population = self.init_population()
        self.best_arch = self.population[0]

        for model in self.population:
            m = self.dict_to_model(model)
            model["fitness"], _ = self.f_objective(m)

            if model['fitness'] >= self.best_arch['fitness']:
                self.best_arch = model

    def crossover(self, target, mutant):
        '''
            Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        return offspring

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.DIMENSIONS) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.DIMENSIONS)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def get_candidates(self,
                       vector, 
                       num=30):

        candidates = []
        # set up hash map
        for d in self.data:
            h = d['hash']
            self.dic[h] = 1

        
        operators = {"rand1": [self.mutation_rand1, 3],
                  "rand2": [self.mutation_rand2, 5],
                  "currentobest1": [self.mutation_currenttobest1, 2],
                  "randtobest2": [self.rand_to_best_2, 5]}
        
        """
        operators = {"currentobest1": [self.mutation_currenttobest1, 2],
                  "randtobest2": [self.rand_to_best_2, 5]}
        """
        # mutate architectures with the lowest loss
        best_arches = sorted(self.population, key=lambda x: x['fitness'], reverse=True)

        # stop when candidates is size num
        # use patience_factor instead of a while loop to avoid long or infinite runtime
        stop_condition = True

        while stop_condition:
            for method, params in operators.items():
                
                target = copy.deepcopy(vector)
                selected_vectors = (i['chromosome'] for i in self.sample_population(self.population, params[1]))

                mutant = None
                if method == 'currentobest1':
                    mutant = params[0](target, best_arches[0]['chromosome'], *(selected_vectors))
                elif method == 'randtobest2':
                    mutant = params[0](best_arches[0]['chromosome'], *(selected_vectors))
                else:
                    mutant = params[0](*(selected_vectors))
                
                trial = self.crossover(target, mutant)
                trial = self.boundary_check(trial)
                config = self.vector_to_config(trial)

                mutated = Model(trial, config, self.CELLS[config[-2]], self.FILTERS[config[-1]], compile=False)
                mutated.encoding = mutated.encode(predictor_encoding='trunc_path', cutoff=self.cutoff)
                
                h = mutated.get_hash()

                if (mutated.isFeasible) and (h not in self.dic):
                    self.dic[h] = 1 
                    candidates.append(self.get_model_dict(mutated, h))

                if len(candidates) >= num:
                    stop_condition = False
                    break  
        
        return candidates

    def evolve_generation(self, predictor=None, num_ensemble = 5):
        query = self.pop_size

        while query <= 160:

            Pnext = []
            generation_candidates = [] # Store all candidates in current generation
            
            for sol in self.population:
                target = sol['chromosome']
                # Generate candidates of target
                candidates = self.get_candidates(target)
                generation_candidates.extend(candidates)

            xtrain = np.array([d['encoding'] for d in self.data])
            ytrain = np.array([(1 - d['fitness']) * 100 for d in self.data])

            xcandidates = np.array([c['encoding'] for c in generation_candidates])
            candidate_predictions = []

            # train an ensemble of neural networks
            train_error = 0

            if predictor == 'bananas':
                for e in range(num_ensemble):
                    meta_neuralnet = MetaNeuralnet()
                    net_params = get_params('meta_standard')['ensemble_params'][e]

                    train_error += meta_neuralnet.fit(xtrain, ytrain, **net_params)

                    # predict the validation loss of the candidate architectures
                    candidate_predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))
                    tf.compat.v1.reset_default_graph()

                train_error /= num_ensemble
                print('query {}, Neural predictor train error: {}'.format(query, train_error))
            
            elif predictor == 'rfc':
                clf = RandomForestRegressor(min_samples_split=5)
                clf.fit(xtrain, ytrain)
                train_error = np.mean(abs(clf.predict(xtrain)-ytrain))
                # predict the validation loss of the candidate architectures
                candidate_predictions.append(np.squeeze(clf.predict(xcandidates)))
                print("MAE:", train_error)
            
            elif predictor == 'xgboost':
                xgb_r = xg.XGBRegressor()
                xgb_r.fit(xtrain, ytrain)
                train_error = np.mean(abs(xgb_r.predict(xtrain)-ytrain))
                # predict the validation loss of the candidate architectures
                candidate_predictions.append(np.squeeze(xgb_r.predict(xcandidates)))
                print("MAE XGBoost:", train_error)

            # compute the acquisition function for all the candidate architectures
            candidate_indices = acq_fn(candidate_predictions, ytrain=ytrain, explore_type=self.explore_type)

            # add the k arches with the minimum acquisition function values
            counter = 0
            for i in candidate_indices:
                candidate_model = self.dict_to_model(generation_candidates[i])
                candidate_model.solNo = self.solNo
                candidate_model.compile_model()
                
                fitness, _ = self.f_objective(candidate_model)
                if candidate_model.isFeasible and fitness != -1:
                    candidate_model.fitness = fitness

                    candidate_model = self.get_model_dict(candidate_model, candidate_model.get_hash())
                    self.data.append(candidate_model)
                    Pnext.append(candidate_model)

                    # we just finished performing k queries
                    query += 1
                    counter += 1
                    self.solNo += 1

                del candidate_model

                if counter >= self.pop_size:
                    break

            del candidates
            tf.keras.backend.clear_session()

            top_5_loss = sorted([(1 - d['fitness']) * 100 for d in self.data])[:min(5, len(self.data))]
            print('{}, query {}, top 5 losses {}'.format(predictor, query, top_5_loss))

            self.population.extend(Pnext)
            self.population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)[:self.pop_size]



    def start_bananas_de(self, num_ensemble = 5, predictor = 'bananas'):
        self.init_rnd_nbr_generators()

        self.init_eval_pop()
        self.data = copy.deepcopy(self.population)
        self.evolve_generation(predictor=predictor)
    
if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")
    device = torch.device('cuda')
    
    data_path = "DataSets/DRIVE"
    batch_size = 128
    seed = 42

    train_dataset = vessel_dataset(data_path, mode="training", split=0.9, de_train=True)
    val_dataset = vessel_dataset(data_path, mode="training", split=0.9, is_val=True, de_train=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    
    print("Number of training image:", train_dataset.__len__())

    path = f"bananas_de_{seed}"
    if not os.path.exists(f"results/{path}/"):
        os.makedirs(f"results/{path}/")


    loss_fn = DiceLoss()
    metric_fn = DiceCoef()

    bananas = BANANAS(explore_type='its')
    bananas.seed_torch(seed)
    bananas.start_bananas_de(predictor='bananas')
