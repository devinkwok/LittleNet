from os.path import join, basename, splitext
from glob import glob

source_dir = './'
target_dir = './article'
do_print = True

print_func = lambda *args, **kwargs: None
if do_print:
    print_func = print
for source_name in glob(join(source_dir, 'experiment-*.py')):
    with open(source_name, 'r+') as source:
        lines = source.readlines()
        print_func('Read file', source_name, len(lines), 'lines')
        target_name = join(target_dir, splitext(basename(source_name))[0] + '.md')
        with open(target_name, 'w+') as target:
            while lines[0] == '\n' or lines[0] == '"""\n':
                lines = lines[1:]
            while lines[-1] == '\n' or lines[-1] == '"""\n':
                lines = lines[:-1]
            is_code_block = False
            for input_line in lines:
                output_line = input_line
                if input_line == '"""\n':
                    if is_code_block:
                        output_line = '```\n'
                        is_code_block = False
                    else:
                        output_line = '```python\n'
                        is_code_block = True
                print_func('... writing:', output_line, end='')
                target.write(output_line)
