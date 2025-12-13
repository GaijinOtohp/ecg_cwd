class CircularQueue:

    def __init__(self, capacity: int = 1) -> None:
        self._capacity: int = capacity
        self._last_node_index: int = -1

        self._nodes_list: list = []


    def enqueue(self, new_node_val):
        # Update last node index
        self._last_node_index = (self._last_node_index + 1) % self._capacity

        # Insert or replace value in the underlying list
        if len(self._nodes_list) < self._capacity:
            self._nodes_list.append(new_node_val)
        else:
            self._nodes_list[self._last_node_index] = new_node_val

        return new_node_val
    

    def get_first(self):
        if len(self._nodes_list) != 0:
            return self._nodes_list[(self._last_node_index + 1) % len(self._nodes_list)]
        return None
    

    def get_last(self):
        if len(self._nodes_list) != 0:
            return self._nodes_list[self._last_node_index]
        return None