from abc import ABC, abstractmethod

class RulesParserInterface(ABC):
    @abstractmethod
    def parse_rule(self, str) -> str:
        pass

    