import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import data.dataBase
import data.loadData
import data.preprocess
import data.postprocess
