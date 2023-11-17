from asyncio import Queue, Lock


class Receiver:
    """
    Abstract class used to reduce communication cost. Selected nodes will become receivers and will only be queried
    for their shares / results / ... on demand (i.e. when byzantine values are detected).
    """

    def __init__(self, max_results, receive_next: bool, node_id) -> None:
        self.receive_in_progress = Queue(maxsize=1)
        self.result_queue = {}
        self.result_queue_lock = Lock()
        self.value_buffer = {}
        self.max_results = max_results
        self.results_received = {}
        self.receive_next = receive_next
        # self.own_value_queue = {}
        # self.own_value_queue_lock = Lock()

    async def await_entry(self):
        await self.receive_in_progress.put(True)
        await self.receive_in_progress.get()

    async def await_result(self, own_value, byzantine, iteration):
        await self.receive_in_progress.put(True)
        # async with self.own_value_queue_lock:
        #     if iteration not in self.own_value_queue:
        #         self.own_value_queue[iteration] = Queue(maxsize=1)
        # await self.own_value_queue[iteration].put((own_value, byzantine))
        async with self.result_queue_lock:
            if iteration not in self.result_queue:
                self.result_queue[iteration] = Queue(maxsize=1)
        res = await self.result_queue[iteration].get()
        self.result_queue.pop(iteration)
        # self.own_value_queue.pop(iteration)
        await self.receive_in_progress.get()
        return res

    async def receive(self, value, byzantine, receive_next: bool, iteration):
        if iteration not in self.value_buffer:
            self.value_buffer[iteration] = []
        self.value_buffer[iteration].append((value, receive_next, byzantine))
        if iteration not in self.results_received:
            self.results_received[iteration] = 0
        self.results_received[iteration] += 1
        if self.results_received[iteration] == self.max_results:
            values = self.value_buffer[iteration].copy()
            async with self.result_queue_lock:
                if iteration not in self.result_queue:
                    self.result_queue[iteration] = Queue(maxsize=1)
            await self.result_queue[iteration].put(values)
            self.value_buffer.pop(iteration)
            self.results_received.pop(iteration)

    # async def collect(self, iteration):
    #     async with self.own_value_queue_lock:
    #         if iteration not in self.own_value_queue:
    #             self.own_value_queue[iteration] = Queue(maxsize=1)
    #     val = await self.own_value_queue[iteration].get()
    #     await self.own_value_queue[iteration].put(val)
    #     return val


class ReceiverManager:

    def __init__(self, receivers: dict) -> None:
        self.receivers = receivers

    async def await_result(self, own_value, byzantine, iteration, sender_group: str):
        if sender_group not in self.receivers.keys():
            raise ValueError(f'Sender group \'{sender_group}\' unknown.')
        return await self.receivers[sender_group].await_result(own_value, byzantine, iteration)

    async def receive(self, value, byzantine, receive_next: bool, iteration, sender_group: str):
        if sender_group not in self.receivers.keys():
            raise ValueError(f'Sender group \'{sender_group}\' unknown.')
        await self.receivers[sender_group].receive(value, byzantine, receive_next, iteration)

    # async def collect(self, sender_group, iteration):
    #     if sender_group not in self.receivers.keys():
    #         raise ValueError(f'Sender group \'{sender_group}\' unknown.')
    #     return await self.receivers[sender_group].collect(iteration)

    def receive_next(self, sender_group: str):
        if sender_group not in self.receivers.keys():
            raise ValueError(f'Sender group \'{sender_group}\' unknown.')
        return self.receivers[sender_group].receive_next

    def receiver(self, sender_group: str):
        if sender_group not in self.receivers.keys():
            raise ValueError(f'Sender group \'{sender_group}\' unknown.')
        return self.receivers[sender_group]
