class BooleanDataset:
    def __init__(self, boolean_op):
        self.boolean_op = boolean_op
        self.data = []
        self.labels = []
    
    def generate_data(self):
        if self.boolean_op == 'AND' or self.boolean_op == 'OR' or self.boolean_op == 'XOR':
            self.data = [[1,0,0], [1,0,1], [1,1,0], [1,1,1]]    # x0, x1, x2
        elif self.boolean_op == 'NOT':
            self.data = [[1,0], [1,1]]                    # x0, x1
        for i in range(len(self.data)):
            if self.boolean_op == 'AND':
                self.labels.append(self.data[i][1] and self.data[i][2])
            elif self.boolean_op == 'OR':
                self.labels.append(self.data[i][1] or self.data[i][2])
            elif self.boolean_op == 'XOR':
                self.labels.append(self.data[i][1] ^ self.data[i][2])
            elif self.boolean_op == 'NOT':
                self.labels.append(not self.data[i])
        
        self.labels = [-1 if label==0 else label for label in self.labels]
