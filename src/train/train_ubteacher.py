from dotenv import find_dotenv, load_dotenv
import os
import sys

load_dotenv(find_dotenv())

sys.path.append(os.environ.get('PROJECT_ROOT'))
sys.path.append(os.environ.get('PROJECT_TRAIN'))

from detectron2.engine import default_argument_parser
from detectron2.engine import default_setup
from detectron2.config import get_cfg

from ubteacher import add_ubteacher_config

from registration import register_all
register_all()



def setup(args):
    '''
    Create cfgs and perform basic setup
    '''
    cfg = get_cfg()
    add_ubteacher_config()
    cfg.merge_from_file(args.config_file)   # adds semi-supervised learning config
    cfg.merge_from_file(args.opts)          # by default does nothing 
    default_setup(cfg, args)
    return cfg

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    args.config_file = os.path.join('ubteacher', 'configs', 'ssl.yaml')
    cfg = setup(args)




