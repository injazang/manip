
csvs = [open(f'E:/Proposals/data/manip/{i}/index.txt','r') for i in range(5)]
train_csvs = [open(f'E:/Proposals/data/manip/{i}.txt', 'w') for i in range(5)]
valid_csv = open('E:/Proposals/data/manip/valid.txt', 'w')
test_csv = open('E:/Proposals/data/manip/test.txt', 'w')

valid_csv.write('split\tname\tmanip\ttype\tparam\n')
test_csv.write('split\tname\tmanip\ttype\tparam\n')

for i in range(5):
    csv = csvs[i]
    lines = csv.readlines()[1:]
    train_csv = train_csvs[i]
    train_csv.write('split\tname\tmanip\ttype\tparam\n')
    for j in range(len(lines)):
        data = lines[j].split(' ')
        if j <77000:
            train_csv.write(f'{i}\t{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}')

        elif j<78000:
            valid_csv.write(f'{i}\t{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}')

        else:
            test_csv.write(f'{i}\t{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}')


