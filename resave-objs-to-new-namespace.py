"""Script for fixing broken imports when pickling and unpickling objects
due to module renaming"""


import sys, os
from littlenet import utility
from littlenet import neural_net
sys.modules['neural_net'] = neural_net

source = './models_old'
target = './models_new'
subdirs = [x for x in os.walk(source)][0][1]
for dir_name in subdirs:
    print('Entering', dir_name)
    for filename, obj in utility.read_all_objects(dir=os.path.join(source, dir_name)):
        print('... copying', filename)
        newpath = os.path.join(target, dir_name)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        utility.write_object(obj, os.path.join(newpath, filename))
