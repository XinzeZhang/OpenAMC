from taskRecog.dataAug.SoftMask.AugWrapper import *

class IQHardAugTask(HardAugTask):
    def optimizer(self, pop_size, maxCycle, func_data, save_dir, metric = 'val_acc', patience = 10, algo = 'clpso'):
        problem_dims = self.model_opts.hyper.sig_len * 2  # the only different which makes the code implements IQ optimization
        peval = Praticle2fitness()
        peval.setup(data=func_data)
        self.fitness_metric = metric
        starting_perturb = [0.05, 0.1, 0.15, 0.2, 0.25,0.5,0]
        results_pt = os.path.join(save_dir, 'hyper.tuning.pt' )
        if os.path.exists(results_pt):
            func_data.logger.info(f'Loading pre-tuning result from {results_pt}')
            maskcode = torch.load(results_pt)
        else:
            def fitness_function(solution):
                maskcode = torch.empty(problem_dims)

                for i in range(problem_dims):
                    state_i = solution[i]
                    rand = uniform(0, 1)
                    maskcode[i] = 1 if state_i > rand else 0

                maskcode = maskcode.view(2,-1) # the only different which makes the code implements IQ optimization
                peval.reset_config(maskcode, IQ=True)
                results = peval.step()
                fitness = results[metric]

                return fitness

            problem = {
                "fit_func": fitness_function,
                "lb": [0 for i in range(problem_dims)],
                "ub": [1 for i in range(problem_dims)],
                "minmax": "max",
                'verbose': False, # 'log_to': "file", 'log_file': self.logger.handlers[0].baseFilename
            }
            term_dict = {
            "max_early_stop": patience   # after 30 epochs, if the global best doesn't improve then we stop the program
            }

            if algo == 'clpso':
                from mealpy.swarm_based import PSO
                algo_func = PSO.CL_PSO(epoch=maxCycle,pop_size=pop_size)
            elif algo == 'aro':
                from mealpy.swarm_based import ARO
                algo_func = ARO.OriginalARO(epoch=maxCycle,pop_size=pop_size)

            s_solutions = self.create_starting_optimSolutions(starting_perturb= starting_perturb, pop_size=pop_size, problem_dims=problem_dims)

            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):
                algo_func.solve(problem, termination=term_dict, starting_positions=s_solutions, mode='swarm')
                best_solution = algo_func.solution[0]

            maskcode = torch.empty(problem_dims)
            for i in range(problem_dims):
                state_i = best_solution[i]
                rand = uniform(0, 1)
                maskcode[i] = 1 if state_i > rand else 0

            maskcode = maskcode.view(2,-1)
            results_pt = os.path.join(save_dir, 'hyper.tuning.pt')
            torch.save(maskcode, results_pt)
            func_data.logger.info(f'Saving tuning result to {results_pt}')

        return maskcode.tolist()

class IQSoftAugTask(SoftAugTask):
    def optimizer(self, pop_size, maxCycle, func_data, save_dir, metric = 'val_acc', patience = 10, algo = 'clpso'):
        problem_dims = self.model_opts.hyper.sig_len * 2
        peval = Praticle2fitness()
        peval.setup(data=func_data)
        self.fitness_metric = metric
        starting_perturb = [0, 0.05, 0.1, 0.15, 0.2, 0.25,0.5]
        results_pt = os.path.join(save_dir, 'hyper.tuning.pt' )
        if os.path.exists(results_pt):
            func_data.logger.info(f'Loading pre-tuning result from {results_pt}')
            maskcode = torch.load(results_pt)
        else:
            def fitness_function(solution):
                maskcode = torch.empty(problem_dims)

                for i in range(problem_dims):
                    maskcode[i] = solution[i]

                maskcode = maskcode.view(2,-1)
                peval.reset_config(maskcode, IQ=True)
                results = peval.step()
                fitness = results[metric]

                return fitness

            problem = {
                "fit_func": fitness_function,
                "lb": [0 for i in range(problem_dims)],
                "ub": [2 for i in range(problem_dims)],
                "minmax": "max",
                'verbose': False, # 'log_to': "file", 'log_file': self.logger.handlers[0].baseFilename
            }
            term_dict = {
            "max_early_stop": patience   # after 30 epochs, if the global best doesn't improve then we stop the program
            }

            if algo == 'clpso':
                from mealpy.swarm_based import PSO
                algo_func = PSO.CL_PSO(epoch=maxCycle,pop_size=pop_size)
            elif algo == 'aro':
                from mealpy.swarm_based import ARO
                algo_func = ARO.OriginalARO(epoch=maxCycle,pop_size=pop_size)

            s_solutions = self.create_starting_optimSolutions(starting_perturb= starting_perturb, pop_size=pop_size, problem_dims=problem_dims)

            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):
                algo_func.solve(problem, termination=term_dict, starting_positions=s_solutions, mode='swarm')
                best_solution = algo_func.solution[0]

            maskcode = torch.empty(problem_dims)
            for i in range(problem_dims):
                maskcode[i] =  best_solution[i]

            maskcode = maskcode.view(2,-1)
            results_pt = os.path.join(save_dir, 'hyper.tuning.pt')
            torch.save(maskcode, results_pt)
            func_data.logger.info(f'Saving tuning result to {results_pt}')

        return maskcode.tolist()