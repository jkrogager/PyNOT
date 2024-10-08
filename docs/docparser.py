from pynot import main
import argparse


def compile_parser_docs():
    """
    Inspect the parser and grab the help strings of all actions.
    """
    parser = main.main(inspect=True)
    docs = {}
    docs['tasks'] = {}
    docs['usage'] = {}
    for action in parser._actions:
        if action.dest == 'help':
            continue

        if isinstance(action, argparse._SubParsersAction):
            # loop through subparsers:
            for task_name, subaction in action.choices.items():
                docs['tasks'][task_name] = subaction._actions[1:]
                docs['usage'][task_name] = subaction.format_usage()
        else:
            docs[action.dest] = action
    return docs


def format_task_options(task_actions, indent=0):

    html_lines = ['\n\n',
                  '<h2 id="summary"> Overview of parameters </h2>\n'
                  ]

    optional_arguments = [action for action in task_actions if not action.required]
    required_arguments = [action for action in task_actions if action.required]

    if len(required_arguments) > 0:
        elements = ['<dl>']
        for arg in required_arguments:
            if len(arg.option_strings) > 0:
                names = sorted(arg.option_strings)
                arg_name = '<dt>%s' % names[0]
                if len(names) > 1:
                    arg_name += ' (%s)' % names[1]
                arg_name += '</dt>'
            else:
                arg_name = '<dt>%s</dt>' % arg.dest
            elements.append(arg_name)
            elements.append('    <dd>%s</dd>' % arg.help)
        elements.append('</dl>')
        html_lines += elements

    if len(optional_arguments) > 0:
        elements = ['<u> Optional Arguments: </u>']
        elements += ['<dl>']
        for arg in optional_arguments:
            if len(arg.option_strings) > 0:
                names = sorted(arg.option_strings)
                arg_name = '<dt>%s' % names[0]
                if len(names) > 1:
                    arg_name += ' (%s)' % names[1]
                arg_name += ':  %r' % arg.default
                arg_name += '</dt>'
            else:
                arg_name = '<dt>%s</dt>' % arg.dest
            elements.append(arg_name)
            elements.append('    <dd>%s</dd>' % arg.help)
        elements.append('</dl>')
        html_lines += elements

    # Add indentation:
    html_lines = [indent*' ' + line for line in html_lines]
    html = '\n'.join(html_lines)
    return html


def format_task_usage(task_usage, indent=0):
    full_usage = task_usage.replace('\n', '')
    full_usage = full_usage.removeprefix('usage: ')
    full_usage = ' '.join([item for item in full_usage.split() if item])
    items = full_usage.split()
    only_positionals = []
    for item in items:
        if ']' in item or '[' in item:
            continue
        else:
            only_positionals.append(item)
    usage = ' '.join(only_positionals)

    elements = [
        '',
        '',
        '<h2> Example Syntax</h2>\n',
        '<h2 class="code">',
        usage,
        '</h2>',
        '',
        '<br>',
        '',
        'Full example of command line syntax: ',
        '<p class="code">',
        full_usage,
        '</p>',
        '<br><br>',
    ]
    elements = [indent*' ' + line for line in elements]
    html = '\n'.join(elements)
    
    return html
