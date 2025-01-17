from minutes import *
from tickets import *
from scipy.sparse import lil_matrix

def test_ticket_spec():
    """Defines key behavior of Tickets class."""
    t = Tickets()
    assert t.N()==0
    assert isinstance(t.N(), int)
    assert isinstance(t.tickets, list)
    assert isinstance(t.tickets_sparse, lil_matrix)
    assert t.tickets_sparse.size==0
    assert t.tickets_sparse.shape==(60, 1440)
    t.add([1, 0])
    assert t.N()==60
    assert all(isinstance(i[0], np.int32) \
               for i in t.tickets_sparse.nonzero())