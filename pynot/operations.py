import os
import sys

from pynot.images import FitsImage, imshift, resample, image_mean, image_log, image_median

sys.tracebacklimit = 1

BLACKLIST = ['import', 'eval', 'rm', 'sudo', 'sh']
functors = {
    'shift': imshift,
    'resample': resample,
    'mean': image_mean,
    'median': image_median,
    'log': image_log,
}

def perform_operation(sequence, variables, output='output.fits'):
    print("\nRunning task: Arithmetic Operator")
    if any(word in sequence.split() for word in BLACKLIST):
        print(" [ERROR]  - Invalid word in the operation. Cannot execute the task!")
        return

    try:
        result = eval(sequence, variables, functors)
        print(f"          - Result is: {result}")
        if isinstance(result, FitsImage):
            result.write(output)
            print(f" [OUTPUT] - Saved result to file: {output}")
        return result

    except Exception:
        print("")
        print("  [ERROR] - Something went wrong!")
        print("            Check that all variables are defined and that all files exist")
        print("")
        raise


def prepare_variables(args_list):
    variables = {}
    errors = []
    messages = []
    for arg in args_list:
        if not '=' in arg:
            errors.append(f" - Unrecognized argument: {arg} - skipping")
            continue
        
        items = arg.split('=')
        if len(items) != 2:
            errors.append(f" - Unrecognized argument: {arg} - skipping")
            continue

        name, value = items
        if name.lower() == 'output':
            variables['output'] = value

        elif os.path.isfile(value):
            variables[name] = FitsImage.read(value)
            messages.append(f" + Assigned variable: {name} = {variables[name]}")

        else:
            try:
                variables[name] = float(value)
                messages.append(f" + Assigned constant: {name} = {variables[name]}")
            except ValueError:
                errors.append(f" - Invalid variable type for {name} = {value}")

    if len(messages) > 0:
        print("   PyNOT operate parser   ")
        print(" ------------------------ ")
        print("\n".join(messages))

    if len(errors) > 0:
        print("\nFollowing errors or warnings ocurred:")
        print("\n".join(errors))
    return variables
