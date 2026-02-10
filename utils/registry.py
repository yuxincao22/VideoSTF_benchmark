class AdapterRegistry:
    def __init__(self):
        self._items = {}

    def register(self, name, cls):
        if name in self._items:
            raise ValueError(f"Adapter already registered: {name}")
        self._items[name] = cls

    def get(self, name):
        if name not in self._items:
            raise KeyError(f"Unknown adapter: {name}. Available: {sorted(self._items.keys())}")
        return self._items[name]

    def names(self):
        return sorted(self._items.keys())


REGISTRY = AdapterRegistry()
