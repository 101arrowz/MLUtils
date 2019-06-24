import pandas as pd
import numpy as np
import warnings
def load(d, literal_data=False, encoding=None, delim=',', dtype=np.float32, header=True, class_col=-1, replace={}, random_seed=np.random.randint(65535), test_split=0.2):
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


    """
    from io import StringIO
    def loadLiteral(d, encoding):
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
    if not isinstance(replace, dict):
        raise TypeError('replace must be a dict with the keys being the strings for replacement and the values being the new strings')
    if not all(isinstance(v, str) and isinstance(k, str) for k,v in replace.items()):
        raise TypeError('every key and value in replace must be a string')
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
    if strindices:
        labels = df[[*class_col]]
        df.drop([*class_col], axis=1, inplace=True)
    else:
        labels = pd.concat([df.iloc[:, col] for col in class_col], axis=1)
        df.drop(df.columns[[*class_col]], axis=1, inplace=True)
    try:
        data = df.to_numpy(dtype=dtype)
        labels = np.squeeze(labels.to_numpy(dtype=dtype))
    except TypeError:
        raise TypeError('invalid dtype: {} is not a legal datatype for a numpy array'.format(dtype)) from None
    np.random.seed(random_seed); np.random.shuffle(data)
    np.random.seed(random_seed); np.random.shuffle(labels)
    lim = int(len(labels)*test_split)
    if not lim:
        raise ValueError("no testing samples. This could be due to improper loading of the data (for example, incorrect delimiter).")
    return (data[:-lim], data[-lim:], labels[:-lim], labels[-lim:])
if __name__ == "__main__":
    print(load('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', dtype=None))