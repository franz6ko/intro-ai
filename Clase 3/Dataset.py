import pickle
import csv
import numpy as np


class Dataset:
    instance = None
    data = None

    def __new__(cls, filename):
        if Dataset.instance is None:
            print("Creating new Dataset instance")
            Dataset.instance = super(Dataset, cls).__new__(cls)
            return Dataset.instance
        else:
            return Dataset.instance

    def __init__(self, filename):
        print("Initialising Dataset")

        try:
            with open(filename + '.pkl', 'rb') as pkl_file:
                self.data = pickle.load(pkl_file)
        except FileNotFoundError:
            print("CSV file found. Building PKL file...")
            try:
                with open(filename + '.csv') as csv_file:
                    with open(filename + '.pkl', 'wb') as pkl_file:

                        csv_reader = csv.reader(csv_file, delimiter=',')

                        def generator(reader):
                            first_skipped = False
                            for line in reader:
                                if not first_skipped:
                                    first_skipped = True
                                    continue
                                yield line[1], line[2]

                        gen = generator(csv_reader)

                        # Simplified solution (Lautaro)
                        #                         gen = ((int(line.split(',')[0]),int(line.split(',')[1]),
                        #                               float(line.split(',')[2]),int(line.split(',')[3]))
                        #                                for i, line in enumerate(csv_file) if i != 0)

                        structure = [('income', np.float32),
                                     ('happiness', np.float32)]

                        array = np.fromiter(gen, dtype=structure)

                        pickle.dump(array, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

                    pkl_file.close()

                with open(filename + '.pkl', 'rb') as pkl_file:
                    self.data = pickle.load(pkl_file)
            except FileNotFoundError:
                print("No PKL or CSV named " + filename + " was found.")
            finally:
                csv_file.close()
        finally:
            pkl_file.close()

    @staticmethod
    def split_dataset(x, y=None, training_percentage=0.7, validation_percentage=None):
        idx = np.random.permutation(x.shape[0])
        x = x[idx]
        y = y[idx]
        a = round(x.shape[0] * training_percentage)

        if validation_percentage is None:
            tx = x[:a]
            sx = x[a:]
            if y is None:
                return tx, sx
            else:
                ty = y[:a]
                sy = y[a:]
                return tx, ty, sx, sy
        else:
            b = round(x.shape[0] * (training_percentage + validation_percentage))
            tx = x[:a]
            vx = x[a:b]
            sx = x[b:]
            if y is None:
                return tx, vx, sx
            else:
                ty = y[:a]
                vy = y[a:b]
                sy = y[b:]
                return tx, ty, vx, vy, sx, sy
