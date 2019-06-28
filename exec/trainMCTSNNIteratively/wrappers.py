def getStateFromNode(node):
    return list(node.id.values())[0]


class GetApproximateValueFromNode:
    def __init__(self, getStateFromNode, getApproximateValue):
        self.getStateFromNode = getStateFromNode
        self.getApproximateValue = getApproximateValue

    def __call__(self, node):
        state = self.getStateFromNode(node)
        approximateValue = self.getApproximateValue(state)

        return approximateValue