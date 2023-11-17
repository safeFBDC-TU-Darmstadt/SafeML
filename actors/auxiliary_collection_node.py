import ray

import configuration as cfg
import util
from util import tensor_function

cpu_res = cfg.threads_available / util.constants.total_nodes


@ray.remote(num_cpus=(cpu_res if cpu_res < 1 else int(cpu_res)))
class Recollector:

    async def collect(self, shapes, length):
        rdm_fct = tensor_function('rand')
        result = []
        for shape in shapes:
            for i in range(length):
                result.append(rdm_fct(shape))
        return result
