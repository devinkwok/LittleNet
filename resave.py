import sys, os
from littlenet import utility
from littlenet import neural_net
sys.modules['neural_net'] = neural_net

source = './models_old'
target = './models_new'
subdirs = [x for x in os.walk(source)][0][1]
for dir_name in subdirs:
    print(dir_name)
    for name, obj in utility.read_all_objects(dir=source + '/' + dir_name):
        print(name)
        newpath = target + '/' + dir_name + '/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        utility.write_object(obj, newpath + name)