from data.RML import (
    MIMO_Nt16Nr4_Data,
    MIMO_Nt4Nr2_Data,
    MIMO_Nt64Nr16_Data,
    HisarMod2019_1_Data,
    Panoradio_HF_Data,
    RML2016_04c_Data,
    RML2016_10a_Data,
    RML2016_10b_Data,
    RML2018_01a_Data,
    RML2022_01a_Data,
    ACMR_Data,
    RML24_Data,
)

# data_zoo = dict(
#     rml16a = ('data.RML', 'RML2016_10a_Data'),
#     rml16b = ('data.RML', 'RML2016_10b_Data'),
#     rml16c = ('data.RML', 'RML2016_04c_Data'),
#     rml18a = ('data.RML', 'RML2018_01a_Data'),
#     a = ('data.RML', 'RML2016_10a_Data'),
#     b = ('data.RML', 'RML2016_10b_Data'),
#     c = ('data.RML', 'RML2016_04c_Data'),
#     p = ('data.RML', 'Panoradio_HF_Data'),
#     dr2 = ('data.RML', 'MIMO_Nt4Nr2_Data'),
#     dr4 = ('data.RML', 'MIMO_Nt16Nr4_Data'),
#     dr16 = ('data.RML', 'MIMO_Nt64Nr16_Data'),
#     h = ('data.RML', 'HisarMod2019_1_Data')

# )


data_zoo = dict(
    rml16a = dict(data_class=RML2016_10a_Data, data_name='RML2016.10a'),
    rml16b = dict(data_class=RML2016_10b_Data, data_name='RML2016.10b'),
    rml16c = dict(data_class=RML2016_04c_Data, data_name='RML2016.04c'),
    rml18a = dict(data_class=RML2018_01a_Data, data_name='RML2018.01a'),
    r = dict(data_class=ACMR_Data, data_name='ACMR'),
    l = dict(data_class=RML24_Data, data_name='RML24'),
    a22 = dict(data_class=RML2022_01a_Data, data_name='RML2022.01a'),
    a = dict(data_class=RML2016_10a_Data, data_name='RML2016.10a'),
    b = dict(data_class=RML2016_10b_Data, data_name='RML2016.10b'),
    c = dict(data_class=RML2016_04c_Data, data_name='RML2016.04c'),
    p = dict(data_class=Panoradio_HF_Data, data_name='Panoradio.HF'),
    dr2 = dict(data_class=MIMO_Nt4Nr2_Data, data_name='MIMO.Nt4Nr2'),
    dr4 = dict(data_class=MIMO_Nt16Nr4_Data, data_name='MIMO.Nt16Nr4'),
    dr16 = dict(data_class=MIMO_Nt64Nr16_Data, data_name='MIMO.Nt64Nr16'),
    h = dict(data_class=HisarMod2019_1_Data, data_name='HisarMod2019.1')
)


def load_data_class(data_name):
    """
    Load a data class based on the dataset name.

    This function dynamically imports and returns the appropriate data class
    for handling different radio modulation datasets (RML2016.10a, RML2016.10b, RML2018.01a).

    Args:
        data_name (str): Name of the dataset. Supported values are:
            - 'rml16a' or 'a': RML2016.10a dataset
            - 'rml16b' or 'b': RML2016.10b dataset
            - 'rml18a': RML2018.01a dataset
            - 'p': Panoradio HF dataset
            - 'dr2': MIMO Nt4Nr2 dataset
            - 'dr4': MIMO Nt16Nr4 dataset
            - 'dr16': MIMO Nt64Nr16 dataset
            - 'h': HisarMod2019_1 dataset

    Returns:
        class: The corresponding data class (e.g., RML2016_10a_Data, RML2016_10b_Data, etc.)
               that can be instantiated to handle the specified dataset.

    Raises:
        Exception: If the provided data_name is not supported in the data_zoo.

    Example:
        >>> data_class = load_data_class('rml16a')
        >>> data_instance = data_class(args)
    """
    if data_name not in data_zoo:
        raise Exception('Unspported attack algorithm {}'.format(data_name))
    return data_zoo[data_name]['data_class']