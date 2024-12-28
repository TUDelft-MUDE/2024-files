from minutes import *
from scipy.sparse import lil_matrix

def test_minutes_spec():
    """Defines key behavior of Minutes class."""
    try_this = Minutes.get_minutes([1, 1, 1])
    assert type(try_this)==list
    assert len(try_this)==1
    try_this = Minutes.get_minutes([1])
    assert type(try_this)==list
    assert len(try_this)==1440
    assert all(isinstance(i, int) for i in try_this)
    sparse_list = Minutes.sparse_list_construct()
    assert isinstance(sparse_list, lil_matrix)
    sparse_list = Minutes.sparse_list_add(sparse_list, try_this)
    assert isinstance(sparse_list, lil_matrix)
    assert sparse_list.size==1440
    assert all(isinstance(i[0], np.int32) for i in sparse_list.nonzero())


def test_single_minutes():
    assert Minutes.get_minutes([1, 1, 1])==[61]
    assert Minutes.get_minutes([1, 0, 0])==[0]
    assert Minutes.get_minutes(['April', 1, 0, 0])==[0]
    assert Minutes.get_minutes([2, 0, 0])==[1440]
    assert Minutes.get_minutes(['April', 2, 0, 0])==[1440]
    assert Minutes.get_minutes([2, 0, 1])==[1441]
    assert Minutes.get_minutes(['April', 2, 0, 1])==[1441]
    assert Minutes.get_minutes([2, 1, 1])==[1501]
    assert Minutes.get_minutes(['April', 2, 1, 1])==[1501]
    may_1 = 30*1440
    assert Minutes.get_minutes([31, 0, 0])==[may_1]
    assert Minutes.get_minutes([30, 23, 59])==[may_1 - 1]
    assert Minutes.get_minutes(['May', 1, 0, 0])==[may_1]

def test_simple_lists():
    assert len(Minutes.get_minutes([1, 1, 1]))==1
    one_hour = Minutes.get_minutes([1, 0])
    assert len(one_hour)==60
    assert min(one_hour)==0
    assert max(one_hour)==59
    one_day = Minutes.get_minutes([1])
    assert len(one_day)==1440
    assert min(one_day)==0
    assert max(one_day)==1439

def test_mult_day_hour_min():
    assert Minutes.get_minutes([[2, 4, 6], 0, 0])==[1440,
                                                    1440 + 1440*2,
                                                    1440 + 1440*4]
    assert Minutes.get_minutes(['April', [2, 4, 6], 0, 0])==[1440,
                                                             1440 + 1440*2,
                                                             1440 + 1440*4]
    assert Minutes.get_minutes([2, [0, 1], 1])==[1441, 1501]
    assert Minutes.get_minutes(['April', 2, [0, 1], 1])==[1441, 1501]
    assert Minutes.get_minutes([2, 0, [1, 3, 5]])==[1441, 1443, 1445]
    assert Minutes.get_minutes(['April', 2, 0, [1, 3, 5]])==[1441, 1443, 1445]
    assert Minutes.get_minutes(['May', 2, 0, [1, 3, 5]])==[1441 + 30*1440,
                                                           1443 + 30*1440,
                                                           1445 + 30*1440]
    
def test_inclusive_day_hour_min():
    assert Minutes.get_minutes([[2, 6], 0, 0])==[1440,
                                                 1440 + 1440*1,
                                                 1440 + 1440*2,
                                                 1440 + 1440*3,
                                                 1440 + 1440*4]
    assert Minutes.get_minutes(['April', [2, 6], 0, 0])==[1440,
                                                             1440 + 1440*1,
                                                             1440 + 1440*2,
                                                             1440 + 1440*3,
                                                             1440 + 1440*4,]
    assert Minutes.get_minutes([2, [0, 3], 1])==[1441, 1501, 1561, 1621]
    assert Minutes.get_minutes(['April', 2, [0, 3], 1])==[1441, 1501, 1561, 1621]
    assert Minutes.get_minutes([2, 0, [1, 5]])==[1441, 1442, 1443, 1444, 1445]
    assert Minutes.get_minutes(['April', 2, 0, [1, 5]])==[1441, 1442, 1443, 1444, 1445]
    assert Minutes.get_minutes(['May', 2, 0, [1, 5]])==[1441 + 30*1440,
                                                        1442 + 30*1440,
                                                        1443 + 30*1440,
                                                        1444 + 30*1440,
                                                        1445 + 30*1440]
    
def test_two_specific_day_hour_min():
    assert Minutes.get_minutes([[2, 6, 6], 0, 0])==[1440,
                                                    1440 + 1440*4]
    assert Minutes.get_minutes(['April', [2, 6, 6], 0, 0])==[1440,
                                                             1440 + 1440*4]
    assert Minutes.get_minutes([2, [0, 3, 3], 1])==[1441, 1621]
    assert Minutes.get_minutes(['April', 2, [0, 3, 3], 1])==[1441, 1621]
    assert Minutes.get_minutes([2, 0, [1, 5, 5]])==[1441, 1445]
    assert Minutes.get_minutes(['April', 2, 0, [1, 5, 5]])==[1441, 1445]
    assert Minutes.get_minutes(['May', 2, 0, [1, 5, 5]])==[1441 + 30*1440,
                                                           1445 + 30*1440]