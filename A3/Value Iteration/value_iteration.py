import numpy as np

value = np.zeros(101)
value[0] = 0
value[100] = 1
beta = 0.000000001
theta = 1
pheads = 0.4

def save_results(data, data_size, filename): # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))

def calculate(action):
    a = min(action, 100-action)
    v = np.zeros(a)
    for i in range(0,a):
        v[i] = pheads*value[action+a] + (1-pheads)*value[action-a]
    return max(v)

def main():
    alpha = 100
    while alpha > beta:
        alpha = 0
        for s in range(1,100):
            v = value[s]
            value[s] = calculate(s)
            alpha = max(alpha, abs(v-value[s]))
pheads = 0.25
main()
save_results(value, 101, "pheads0.25.dat")
pheads = 0.6
main()
save_results(value, 101, "pheads0.6.dat")
