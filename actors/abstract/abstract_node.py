class AbstractNode:

    def __init__(self, node_id, byzantine, byzantine_dict) -> None:
        self.node_id = node_id
        self.byzantine = byzantine
        self.byzantine_dict = byzantine_dict
