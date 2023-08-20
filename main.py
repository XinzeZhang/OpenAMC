from task.TaskParser import get_parser
from task.base.TaskWrapper import Task

if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    
    args.exp_config = 'exp_config/xinze'
    args.exp_file= 'rml16a'
    args.exp_name = 'rl.test'
    
    args.test = True
    args.clean = True
    
    args.model = 'awn'
    
    
    task = Task(args)
    task.conduct()
    # task.evaluate()
    