import csv

with open('../Datasets/candidateElimination.csv') as f:
    csvFile = csv.reader(f)
    data = list(csvFile)

    s = data[1][:-1]
    g = [['?' for i in range(len(s))] for j in range(len(s))]

    for i in data:
        if i[-1] == 'Yes':
            for j in range(len(s)):
                if i[j] != s[j]:
                    s[j] == '?'
                    g[j][j] = '?'
        elif i[-1] == 'No':
            for j in range(len(s)):
                if i[j] != s[j]:
                    g[j][j] = s[j]
                else:
                    g[j][j] = '?'
        print('Steps of candidate elimination algorithm',data.index(i)+1)
        print(s)
        print(g)
    gh = []
    for i in g:
        for j in i:
            if j != '?':
                gh.append(i)
                break
    print("Final specific hypothesis:\n",s)
    print('Final general Hypothesis:\n',gh)