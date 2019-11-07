from mxnet import np, npx
import sys
import pandas as pd

data_file = '../data/house_tiny.csv'


def read_csv():
    data = pd.read_csv(data_file)
    print(data)


def write_csv():
    # Write the dataset row by row into a csv file
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # Column names
        f.write('NA,Pave,127500\n')  # Each row is a data point
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')


def main():
    # write_csv()
    read_csv()


if __name__ == '__main__':
    main()
    sys.exit(0)
