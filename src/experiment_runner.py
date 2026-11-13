"""Execute independent fits in one process while retaining immutable CPU graphs."""
import gc
import traceback

from src.data_cache import DatasetCache


class ExperimentRunner:
    def __init__(self, cache_dir="./data/cache/graphs"):
        self.data_cache = DatasetCache(cache_dir)

    def run(self, arguments):
        from scripts.run_single import main
        try:
            main(list(arguments), data_cache=self.data_cache)
            return True
        except (Exception, SystemExit):
            traceback.print_exc()
            return False
        finally:
            # Keep cached graphs and the CUDA allocator; discard fit-local objects.
            # run_single resets seeds and creates a fresh model/optimizer each fit.
            gc.collect()
