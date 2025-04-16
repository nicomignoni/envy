from numbers import Number

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

# Helpers for NV.__repr__ 
def array_type_str(shape):
    if len(shape) >= 3:
        return f"Array {shape}"
    elif len(shape) == 2:
        return f"Matrix {shape}"
    elif len(shape) == 1:
        return f"{shape[0]}-elements Vector"
    elif len(shape) == 0:
        return f"Number"

class NV(NDArrayOperatorsMixin):
    def __init__(self, **elements):
        l = 0
        self._layout = dict()
        self._vector = list()
        for key, element in elements.items():
            if isinstance(element, np.ndarray):
                size = element.size
                shape = element.shape
                self._vector.append(element.flatten())
            elif isinstance(element, Number): 
                size = 1
                shape = ()
                self._vector.append([element])
            else:
                raise TypeError(
                    f"Elements can be np.ndarray or Number; {key} is a {type(element)}."
                )
            self._layout[key] = (shape, l, l + size) # shape, start, end
            l += size
        self._vector = np.concatenate(self._vector)

    def __repr__(self):
        layout = "\n".join(
            f"{key}: {array_type_str(shape)} [{start}-{end})" 
            for key, (shape, start, end) in self._layout.items()
        )
        return (
            f"{self.size}-element {self.__class__.__name__} with layout:\n"
            f"{layout}\n{self._vector}"
        )

    def __getitem__(self, key):
        if key in self.layout:
            shape, start, end = self._layout[key]
            return np.reshape(self._vector[start:end], shape)
        else:
            # If it's not a string, it should be a numpy index
            return self._vector[key]

    def __setitem__(self, key, value):
        if key in self.layout:
            shape, start, end = self._layout[key]
            if isinstance(value, np.ndarray):
                # Checking for shape is needed since, e.g., arrays of size 
                # (2,6) and (3,4) have the same flattening. 
                if value.shape == shape:
                    self._vector[start:end] = value.flatten()
                else:
                    raise ValueError(
                        f"Dimension mismatch between shapes {value.shape} and {shape}"
                    )
            elif isinstance(value, Number):
                self._vector[start:end] = value
        else:
            # If it's not a string, it should be a numpy index
            self._vector[key] = value

    # See https://numpy.org/doc/stable/user/basics.dispatch.html
    def __array__(self, dtype=None, copy=None):
        return self._vector.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            parsed_inputs = list()
            layout_mismatch = False
            for input in inputs:
                if isinstance(input, self.__class__):
                    layout_mismatch = layout_mismatch or input._layout != self._layout
                    parsed_inputs.append(input._vector)
                else:
                    # TODO: check for other input types
                    parsed_inputs.append(input)

            # An NV is returned if the arguments NVs have the same size and layout, 
            # otherwise it falls back to whatever array is given by ufunc
            result = ufunc(*parsed_inputs, **kwargs)
            if not layout_mismatch and result.shape == self.shape:
                nv = self.__new__(NV)
                nv._layout = self._layout
                nv._vector = result
                return nv
            else:
                return result
        else:
            NotImplemented

    @property
    def shape(self):
        return self._vector.shape

    @property
    def size(self):
        return self._vector.size

    @property
    def ndim(self):
        return self._vector.ndim

    @property
    def layout(self): 
        return self._layout
