import pandas as pd
import numpy as np
import warnings
import os
from hashlib import md5
import pickle
class DataLoader:
    def __init__(self, d, use_save=False, save=True, literal_data=False, encoding=None, delim=',', dtype=np.float32, header=True, class_col=-1, remove=None, fill_to_max=None, dupes_one_set=True, replace={}, unknown=None, one_hot=False, random_seed=np.random.randint(65535), test_split=0.2):
        """
        A data loader.

        `d`
        A filepath, URL, or raw string to load data from.

        `use_save`
        Whether to use a pickled version of the current dataset. Make "True" when finished data manipulation.

        `save`
        Whether to pickle and save the current dataset.

        `literal_data`
        Whether `d` should be interpreted literally. Optional, but avoids a warning.

        `encoding`
        The data's encoding. The default of `None` will automatically pick the best encoding.

        `delim`
        The delimiter to be passed to `pandas.read_csv`.

        `dtype`
        The datatype of the numpy arrays. Set to `None` to let `pandas` decide or have different datatypes for the labels and data.

        `header`
        Whether there is a header containing labels for each column
        
        `class_col`
        An index/iterable of indices or name/iterable of names for the column(s) containing labels.

        `remove`
        An index/iterable of indices or name/iterable of names for the column(s) to be outright removed.
        
        `fill_to_max`
        Whether or not to equalize the amount of data for each class.

        `dupes_one_set`
        Whether or not to move duplicates all to one set randomly. Always true with `fill_to_max` != None. Strongly suggested to leave True.

        `replace`
        A dictionary with keys (to replace) and values (what to replace with)

        `one_hot`
        Whether to encode the labels as one-hot vectors.

        `seed`
        The random seed. Use to remove randomness.

        `test_split`
        The decimal out of one to use as testing data.
        """

        if __name__ == '__main__': os.chdir(os.path.dirname(os.path.abspath(__file__)))
        fulldata = None
        cachedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__pycache__')
        if not os.path.isdir(cachedir):
            if os.path.exists(cachedir):
                os.remove(cachedir)
            os.makedirs(cachedir)
        if not isinstance(d, (str, bytes)):
            raise Exception('p must be of type string (or bytes when passing raw, encoded data)')
        fn = md5(d if isinstance(d, bytes) else d.encode('UTF-8')).hexdigest()
        datafp = os.path.join(cachedir, fn+'.pkldata')
        if use_save:
            try:
                with open(datafp, 'rb') as f:
                    self.saveData(pickle.load(f))
                    return
            except FileNotFoundError:
                print('Failed to find saved file! Loading directly...')
        fulldata = self.__class__.load(**locals())
        if save:
            try:
                with open(datafp, 'wb') as f:
                    pickle.dump(fulldata, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Could not save data for an unknown reason. See the following exception:', e)
        self.saveData(fulldata)
    def saveData(self, fulldata):
        self.data = fulldata
        self.X_train, self.X_test, self.y_train, self.y_test = self.data
    @staticmethod
    def load(d, literal_data=False, encoding=None, delim=',', dtype=np.float32, header=True, class_col=-1, remove=None, fill_to_max=None, dupes_one_set=True, replace={}, unknown=None, one_hot=False, random_seed=np.random.randint(65535), test_split=0.2, **kwargs):
        """
        Returns a tuple (X_train, X_test, y_train, y_test) given a dataset.

        `d`
        A filepath, URL, or raw string to load data from.

        `literal_data`
        Whether `d` should be interpreted literally. Optional, but avoids a warning.

        `encoding`
        The data's encoding. The default of `None` will automatically pick the best encoding.

        `delim`
        The delimiter to be passed to `pandas.read_csv`.

        `dtype`
        The datatype of the numpy arrays. Set to `None` to let `pandas` decide or have different datatypes for the labels and data.

        `header`
        Whether there is a header containing labels for each column
        
        `class_col`
        An index/iterable of indices or name/iterable of names for the column(s) containing labels.

        `remove`
        An index/iterable of indices or name/iterable of names for the column(s) to be outright removed.
        
        `fill_to_max`
        Whether or not to equalize the amount of data for each class.

        `dupes_one_set`
        Whether or not to move duplicates all to one set randomly. Always true with `fill_to_max` != None. Strongly suggested to leave True.

        `replace`
        A dictionary with keys (to replace) and values (what to replace with)

        `one_hot`
        Whether to encode the labels as one-hot vectors.

        `seed`
        The random seed. Use to remove randomness.

        `test_split`
        The decimal out of one to use as testing data.
        """
        from io import StringIO
        def loadLiteral(d, encoding):
            raw = None
            try:
                if encoding:
                    raw = d.decode()
                else:
                    raw = d
            except AttributeError:
                raise ValueError("cannot have encoding argument when passing data as an unencoded string") from None
            return raw
        if not isinstance(d, (str, bytes)):
            raise TypeError("d must be of type string (or bytes when passing raw, encoded data)")
        if not isinstance(literal_data, bool):
            raise TypeError("literal_data must be of type string or bytes (when passing raw, encoded data)")
        if not isinstance(encoding, str) and encoding is not None:
            raise TypeError('encoding must be of type string or None')
        if not isinstance(header, bool):
            raise TypeError('header must be of type bool')
        if isinstance(class_col, (int, str)):
            class_col = (class_col,)
        strindices = False
        try:
            if not all(map(lambda v: isinstance(v, int), class_col)):
                if not all(map(lambda v: isinstance(v, str), class_col)):
                    raise TypeError('class_col must contain only int values or only str values')
                else:
                    strindices = True
                    if not header:
                        raise TypeError("a header must exist to access class_col with str indices")
        except TypeError as e:
            raise (TypeError('class_col must be an int or an iterable') if str(e)[0] == "'" else e) from None
        if isinstance(remove, (int, str)):
            remove = (remove,)
        strindices = False
        if remove is not None:
            try:
                if not all(map(lambda v: isinstance(v, int), class_col)):
                    if not all(map(lambda v: isinstance(v, str), class_col)):
                        raise TypeError('remove must contain only int values or only str values')
                    else:
                        strindices = True
                        if not header:
                            raise TypeError("a header must exist to access remove with str indices")
            except TypeError as e:
                raise (TypeError('remove must be an int or an iterable') if str(e)[0] == "'" else e) from None
        if not isinstance(fill_to_max, bool):
            if fill_to_max is not None:
                raise TypeError('fill_to_max must be a boolean or None (if no replication of terms is desired)')
        elif len(class_col) < 1:
            raise ValueError('cannot have multiple class columns if filling. Set fill_to_max to None to remove this error.')
        if not isinstance(dupes_one_set, bool):
            raise TypeError('dupes_one_set must be of type bool')
        if not isinstance(one_hot, bool):
            raise TypeError('one_hot must be of type bool')
        if not isinstance(replace, dict):
            raise TypeError('replace must be a dict with the keys being the strings for replacement and the values being the new strings')
        if isinstance(unknown, (str)):
            unknown = (unknown,)
        if unknown is not None:
            try:
                if not all(map(lambda v: isinstance(v, str), unknown)):
                    raise TypeError('unknown must contain only str values')
            except TypeError as e:
                raise (TypeError('unknown must be a str or an iterable') if str(e)[0] == "'" else e) from None
        if not all(isinstance(v, str) and isinstance(k, str) for k,v in replace.items()):
            raise TypeError('every key and value in replace must be a string')
        if not isinstance(random_seed, int):
            raise TypeError('random_seed must be of type int')
        if not isinstance(test_split, float):
            raise TypeError('test_split must be of type float')
        if not (0 < test_split < 1):
            raise ValueError('test_split must be greater than 0 but less than 1')
        header = 0 if header else None
        if literal_data:
            raw = loadLiteral(d, encoding)
        else:
            try:
                with open(d, encoding=encoding) as f:
                    raw = f.read()
            except (FileNotFoundError, OSError):
                try:
                    import requests
                    r = requests.get(d)
                    r.encoding = encoding if encoding else r.apparent_encoding
                    raw = r.text
                except:
                    warnings.warn("interpreting d as a literal string of data. This is usually not the best solution. To disable this warning, set literal_data to True.", RuntimeWarning)
                    raw = loadLiteral(d, encoding)
        if not raw:
            raise ValueError("input file or string contained no data")
        df = pd.read_csv(StringIO(raw), sep=delim, header=header)
        for k, v in replace.items(): df.replace(k, v, inplace=True)
        if unknown is not None:
            for s in df:
                not_invalid = df[~df[s].isin(unknown)]
                for val in unknown:
                    df[s].replace(val, not_invalid[s].sample().to_numpy()[0], inplace=True)
        if fill_to_max is not None:
            labels = df[class_col[0]] if strindices else df.iloc[:, class_col[0]]
            valcs = labels.value_counts()
            if fill_to_max:
                maxc = max(valcs)
                for c in labels.unique():
                    subset = df[labels == c]
                    a = valcs[c]
                    diff = maxc-a
                    for i in range(diff//a):
                        df = df.append(subset, ignore_index=True)
                    if diff % a:
                        np.random.seed(random_seed); df = df.append(subset.iloc[np.random.choice(len(subset), diff % a, replace=False)])
            else:
                raise NotImplementedError("fill_to_max=False is not supported. Use None for no filling or removal, or True for filling.")
        if strindices:
            labels = df[class_col[0]]
            df.drop([*class_col], axis=1, inplace=True)
            if remove is not None: df.drop([*remove], axis=1, inplace=True)
        else:
            labels = df.iloc[:, class_col[0]]
            df.drop(df.columns[[*class_col]], axis=1, inplace=True)
            if remove is not None: df.drop(df.columns[[*remove]], axis=1, inplace=True)
        try:
            data = df.to_numpy(dtype=dtype)
            labels = np.squeeze(labels.to_numpy(dtype=dtype))
        except TypeError:
            raise TypeError('invalid dtype: {} is not a legal datatype for a numpy array'.format(dtype)) from None
        if one_hot:
            unique_labels = np.unique(labels)
            labels = np.asarray([np.asarray([0 if labels[i] != unique_labels[g] else 1 for g in range(len(unique_labels))]) for i in range(labels.shape[0])])
            # datat = np.transpose(data)
            # unique_vals = [np.unique(datat[n]) for n in range(datat.shape[0])]
            # print(unique_vals[0])
            # data = np.asarray([np.asarray([np.asarray([0 if data[g][b] != unique_vals[b][i] else 1 for i in range(len(unique_vals[b]))]) for b in range(data.shape[1])]) for g in range(data.shape[0])])
        np.random.seed(random_seed); np.random.shuffle(data)
        np.random.seed(random_seed); np.random.shuffle(labels)
        lim = int(len(labels)*test_split)
        if not lim:
            raise ValueError("no testing samples. This could be due to improper loading of the data (for example, incorrect delimiter).")
        fulldata =  (data[:-lim], data[-lim:], labels[:-lim], labels[-lim:])
        # Don't want to touch inner layers
        if fill_to_max or dupes_one_set:
            fulldata = list(map(list, fulldata))
            for el in data: # Removing duplicates
                remfrom, addto = (0, 1) if np.random.rand() < test_split else (1, 0)
                getInd = lambda : [i for i in range(len(fulldata[remfrom])) if all(fulldata[remfrom][i] == el)]
                while getInd():
                    ind = getInd()[0]
                    fulldata[remfrom].pop(ind)
                    fulldata[addto].append(el)
                    v = fulldata[remfrom+2].pop(ind)
                    fulldata[addto+2].append(v)
            for i in range(4):
                fulldata[i] = np.asarray(fulldata[i])
                np.random.seed(random_seed); np.random.shuffle(fulldata[i])
            fulldata = tuple(fulldata)
        return fulldata