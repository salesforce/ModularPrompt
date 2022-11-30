import argparse
import pdb
import numpy as np
from collections import defaultdict

class LParser(object):
    def __init__(self, args):
        self.args = args 
    
    def reprint_test_stats_wild (self, wild_stages):
        '''reprint test stats, following continual learning label classes
        '''
        def _print(data_dict, label_set):
            for la in label_set:
                #stats_lst = map(lambda x: f'{x:<4}', data_dict[la]) # [0.8, 0.7, 0.5] -> ['0.80', '0.70', '0.50']
                delimiter = ',\t'
                la_ = la.replace(' ', '_')
                _line = f"{la_:<25}{delimiter}{delimiter.join(data_dict[la_])}"
                print(_line)
        def _parse(lines):
            data_dict = {}
            for line in lines:
                d = line.split()
                idx = 0
                while(d[idx][0].isalpha()):
                    idx += 1
                label_name = '_'.join(d[0:idx])
                data_dict[label_name] = [d[idx], d[idx+1], d[idx+2]]
            return data_dict
        
        with open(args.logfile, 'r') as rf:
            lines = rf.readlines()
        # iterate lines to acquire data
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if 'data file' in line:
                print('+'*64)
                print(line)
            if 'f1-score' in line and 'support' in line: # start of a test stats
                # read all info of cur test stats
                test_lines = []
                idx += 2 # skip the next line break
                while(lines[idx] != '\n'):
                    test_lines.append(lines[idx])
                    idx += 1
                data_dict = _parse(test_lines)
                for k,v in wild_stages.items():
                    print(f"wild stage {k}:{'-'*48}")
                    _print(data_dict, v)

                idx += 1
                while(lines[idx] != '\n'):
                    print(lines[idx])
                    idx += 1
            idx += 1

    def reprint_test_stats_wild_v2(self, dataset_order, num_runs, wild_resume_stage=0):
        '''Given training log with args.wild_do_test=all_seen, print task-specific, task-agnostic test F1
        Args:
            dataset_order: e.g. ['dbpedia', 'agnews' ,'yahoo']
        '''
        def _var(*args):
            sum = 0
            for a in args:
                if a:
                    sum += np.array(a)
            sum /= len(args)
            return (np.mean(sum), np.std(sum))
        
        def __print(task_test_specific, dataset_order):
            for task_name in dataset_order:
                cur_test = task_test_specific[task_name]
                u, std = _var(cur_test)
                print(f"{task_name}: [{u:.3f} / {std:.3f}] {cur_test}")
            u, std = _var(*list(task_test_specific.values()))
            print(f"[{u:.3f} / {std:.3f}]")
            print(f"{'-'*20}")

        def _print(task_test, dataset_order):
            for num_tasks in range(1, len(dataset_order)+1):
                cur_test = task_test[num_tasks]
                for cur_data in dataset_order:
                    if cur_data in cur_test:
                        u, std = _var(cur_test[cur_data])
                        print(f"{cur_data}: [{u:.3f} / {std:.3f}] {cur_test[cur_data]}")
                u, std = _var(*list(cur_test.values()))
                print(f"[{u:.3f} / {std:.3f}]")
                print(f"{'-'*20}")

        # firstly get all the tested accuracies
        test_accs = []
        with open(args.logfile, 'r') as rf:
            lines = rf.readlines()
        for line in lines:
            if 'accuracy' in line:
                test_accs.append(float(line.split()[1]))
            elif 'micro avg' in line:
                test_accs.append(float(line.split()[4]))
        print(f'all test accuracies: {test_accs}')
        # regroup the accs into task specific, task-agnostic
        task_test_specific = defaultdict(list) # {task_name:[acc1, acc2]}
        task_test = {i:{task_name:[] for task_name in dataset_order[:i]} for i in range(1, len(dataset_order)+1)} # {num_tasks: {task_1:[acc1,acc2],...]}
        acc_index = 0
        for run_id in range(num_runs):
            for num_tasks in range(1, len(dataset_order)+1):
                if wild_resume_stage > 0 and num_tasks <= wild_resume_stage:
                    continue
                # if run_id == 0 and num_tasks == wild_resume_stage + 1:
                #     continue
                # special skip for prompt_wild 
                # acc_index += [0,2,5,9,14][num_tasks-1]
                for cur_task in dataset_order[:num_tasks]:
                    task_test[num_tasks][cur_task].append(test_accs[acc_index])
                    acc_index += 1
                task_test_specific[dataset_order[num_tasks-1]].append(test_accs[acc_index])
                acc_index += 1
                # acc_index += [0,3,2,1,0][num_tasks-1]
            # acc_index += 1 # skip standard test
        
        # print the regrouped test acc
        _print(task_test, dataset_order)
        __print(task_test_specific, dataset_order)

    def reprint_test_stats_wild_v3(self, dataset_order, num_runs):
        '''For each run, test once for each dataset, plus one task specific test, plus one final test
        '''
        def _var(*args):
            sum = 0
            for a in args:
                    sum += np.array(a)
            sum /= len(args)
            return (np.mean(sum), np.std(sum))

        def __print(task_test_specific):
            u, std = _var(task_test_specific)
            print(f"specific: [{u*100:.3f} / {std*100:.3f}] {task_test_specific}")
        
        def _print(task_test, dataset_order):
            for i, task in enumerate(dataset_order):
                cur_test = task_test[task]
                u, std = _var(cur_test)
                print(f"{task}: [{u*100:.3f} / {std*100:.3f}] {cur_test}")
            u, std = _var(*list(task_test.values()))
            print(f"[{u*100:.3f} / {std*100:.3f}]")
            print(f"{'-'*20}")

        # firstly get all the tested accuracies
        test_accs = []
        with open(args.logfile, 'r') as rf:
            lines = rf.readlines()
        for line in lines:
            if 'accuracy' in line:
                test_accs.append(float(line.split()[1]))
            elif 'micro avg' in line:
                test_accs.append(float(line.split()[4]))
        print(f'all test accuracies: {test_accs}')

        task_test = {task_name:[] for task_name in dataset_order} # {task_1:[acc1,acc2],..}
        task_test_specific = [] # {task_name:[acc1, acc2]}
        acc_index = 0
        for run_id in range(num_runs):
            for cur_task in dataset_order:
                task_test[cur_task].append(test_accs[acc_index])
                acc_index += 1
            # task_test_specific.append(test_accs[acc_index])
            # acc_index += 1
            # acc_index += 1 # skip standard test
        # print the regrouped test acc
        _print(task_test, dataset_order)
        # __print(task_test_specific)

    def reprint_test_stats_wild_fused(self, dataset_order, num_runs):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogParser")
    parser.add_argument("--logfile", dest="logfile", type=str,
                        default="387.log", help="path to log file")
    parser.add_argument("--nruns", dest="nruns", type=int,
                        default=5, help="path to log file")
    parser.add_argument("--resume_stage", dest="resume_stage", type=int,
                        default=0, help="wild resume stage")
    parser.add_argument("--current", dest="current", action="store_true",
                        default=False, help="if only test current")
    parser.add_argument("--nstages", dest="nstages", type=int,
                        default=5, help="number of stages")
    args = parser.parse_args()
    lp = LParser(args)
    stages = [str(i) for i in range(args.nstages)]
    if args.current:
        lp.reprint_test_stats_wild_v3(dataset_order=stages, num_runs=args.nruns)
    else:
        lp.reprint_test_stats_wild_v2(dataset_order=stages, num_runs=args.nruns, wild_resume_stage=args.resume_stage)
    # lp.reprint_test_stats_wild (wild_stages={0:["company", "educational institute", "artist", "athlete", "office holder", "means of transportation", "building"], 
    #         1:["natural place", "village", "animal", "plant", "album", "film", "written work"]})
    # lp.reprint_test_stats_wild_v2(dataset_order=['dbpedia', 'agnews'], num_runs=args.nruns, wild_resume_stage=0)
    # lp.reprint_test_stats_wild_v2(dataset_order=['1', '2', '3', '4', '5'], num_runs=args.nruns, wild_resume_stage=args.resume_stage)
    # lp.reprint_test_stats_wild_v3(dataset_order=['1', '2', '3', '4', '5'], num_runs=args.nruns)
    # lp.reprint_test_stats_wild_v2(dataset_order=['1', '2', '3', '4'], num_runs=args.nruns, wild_resume_stage=args.resume_stage)
    # lp.reprint_test_stats_wild_v3(dataset_order=['1', '2', '3', '4'], num_runs=args.nruns)
    # lp.reprint_test_stats_wild_v2(dataset_order=['dbpedia', 'agnews', 'huffpost'], num_runs=args.nruns, wild_resume_stage=0)
    # lp.reprint_test_stats_wild_v2(dataset_order=['huffpost', 'dbpedia', 'agnews'], num_runs=args.nruns, wild_resume_stage=0)
    # lp.reprint_test_stats_wild_v2(dataset_order=['agnews', 'huffpost', 'dbpedia'], num_runs=args.nruns, wild_resume_stage=0)
    # lp.reprint_test_stats_wild_v3(dataset_order=['dbpedia', 'agnews', 'huffpost'], num_runs=args.nruns)
    # lp.reprint_test_stats_wild_v3(dataset_order=['huffpost'], num_runs=args.nruns)
