from task.TaskParser import get_parser
from task.TaskWrapper import Task

if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    
    args.expdatafolder = 'exp_config/xinze'
    args.exp_file= 'rml16a'
    args.exp_name = 'paper.test'
    
    args.test = True
    args.model = 'amcnet'
    
    
    task = Task(args)
    task.conduct()
    task.evaluate()
    