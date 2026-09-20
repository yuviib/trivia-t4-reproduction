import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import sys
sys.path.insert(0, '/kaggle/working/TRivia/training/ms-swift')

from swift.llm import rollout_main

if __name__ == '__main__':
    rollout_main()
