"""
PyNOT : Task Manager

This module handles the parsing of tasks from the options.yml file
"""

from collections import defaultdict

from pynot.logging import Report

valid_tasks = ['bias', 'flat', 'arcs', 'identify', 'response', 'science']

base_task = [{}]
default_tasks = {name: base_task for name in valid_tasks}


class WorkflowParsingError(Exception):
    pass


def parse_tasks(options, log=None, verbose=True):
    if log is None:
        log = Report(verbose)
    tasks = defaultdict(list)
    N_valid = 0
    log.add_linebreak()
    log.write("Preparing data processing workflow")
    if 'workflow' in options and options['workflow'] is not None:
        if not isinstance(options['workflow'], list):
            log.error("Invalid format in workflow! Must specify a list of tasks in the .pfc file:")
            log.write("workflow:", prefix=12*' ')
            log.write("- task_name:", prefix=12*' ')
            log.write("    task_parameter1: value", prefix=12*' ')
            log.write("    task_parameter2: value", prefix=12*' ')
            log.add_linebreak()
            raise WorkflowParsingError("Invalid workflow format")

        for num, task in enumerate(options['workflow'], 1):
            if len(task.keys()) != 1:
                log.warn("Too many arguments of task number %i in current workflow" % num)
                log.warn("%r" % task)
                log.warn("Skipping this task in the workflow...")
                continue
            task_name, pars = task.popitem()
            if task_name in valid_tasks:
                if pars is None:
                    pars = {}
                tasks[task_name].append(pars)
                N_valid += 1
            else:
                log.warn("Invalid task name: %s" % task_name)
                log.warn("Must be one of: %s" % ", ".join(valid_tasks))
                log.warn("Skipping this task in the workflow...")
        N_tasks = len(options['workflow'])
        log.write("Number of tasks passed: %i" % N_tasks)
        log.write("Number of tasks successfully parsed: %i" % N_valid)

    else:
        log.write("No tasks specified in the workflow")
        log.write("Using default workflow:")
        log.write("%s" % ", ".join(valid_tasks))
        tasks = default_tasks
    log.add_linebreak()
    log.add_linebreak()
    return tasks
