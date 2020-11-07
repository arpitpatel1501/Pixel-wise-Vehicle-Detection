##Project Representation:

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference
import networkx as nx
import matplotlib.pyplot as plt

dbn = DBN()

# Vt ,vt-1 , st ,ct,et
#dbn.add_edges_from([(('D', 0),('G', 0)),(('I', 0),('G', 0)),(('D', 0),('D', 1)),(('I', 0),('I', 1))])
#dbn.add_edges_from([(('V(t-1)', 0),('V(t-1)', 1)),(('V(t-1)', 0),('S(t-1)', 0)),(('V(t-1)', 0),('E(t-1)', 0)),(('V(t-1)', 0),('C(t-1)', 0)),(('V(t)', 1),('S(t)', 1)),(('V(t)', 1),('E(t)', 1)),(('V(t)', 1),('C(t)', 1))])
#dbn.add_edges_from([(('V(t)', 0),('V(t)', 1)),(('V(t)', 0),('S(t)', 0)),(('V(t)', 0),('E(t)', 0)),(('V(t)', 0),('C(t)', 0))])
dbn.add_edges_from([(('V(t)', 0),('V(t)', 1)),(('S(t)', 0),('V(t)', 0)),(('E(t)', 0),('V(t)', 0)),(('C(t)', 0),('V(t)', 0))])


Vt_1_cpd = TabularCPD(('V(t)', 0), 2, [[0.506257, 0.493743]])

st_cpd = TabularCPD(('S(t)', 1), 2, [[0.989, 0.658],[0.0103, 0.3412]],evidence=[('V(t)', 0)],evidence_card=[2])

et_cpd = TabularCPD(('E(t)', 1), 2, [[0.9828, 0.4068],[0.017108, 0.59317]],evidence=[('V(t)', 0)],evidence_card=[2])

ct_cpd = TabularCPD(('C(t)', 1), 2, [[0.9388, 0.82414],[0.0611017, 0.17585]],evidence=[('V(t)', 0)],evidence_card=[2])

Vt_cpd = TabularCPD(('V(t)', 1), 2, [[0.96, 0.1],[0.04, 0.9]],evidence=[('V(t)', 0)],evidence_card=[2])

dbn.add_cpds(Vt_1_cpd,Vt_cpd, st_cpd, et_cpd, ct_cpd)

dbn.initialize_initial_state()

dbn_inf = DBNInference(dbn)

#dbn_inf.query([('X', 0)], {('Y', 0):0, ('Y', 1):1, ('Y', 2):1})[('X', 0)].values
#array([ 0.66594382,  0.33405618])

nx.draw(dbn,with_labels=True)
plt.show()


'''

dbnet = DBN()
>>> dbnet.add_edges_from([(('Z', 0), ('X', 0)), (('X', 0), ('Y', 0)),
...                       (('Z', 0), ('Z', 1))])
>>> z_start_cpd = TabularCPD(('Z', 0), 2, [[0.5, 0.5]])
>>> x_i_cpd = TabularCPD(('X', 0), 2, [[0.6, 0.9],
...                                    [0.4, 0.1]],
...                      evidence=[('Z', 0)],
...                      evidence_card=[2])
>>> y_i_cpd = TabularCPD(('Y', 0), 2, [[0.2, 0.3],
...                                    [0.8, 0.7]],
...                      evidence=[('X', 0)],
...                      evidence_card=[2])
>>> z_trans_cpd = TabularCPD(('Z', 1), 2, [[0.4, 0.7],
...                                        [0.6, 0.3]],
...                      evidence=[('Z', 0)],
...                      evidence_card=[2])
>>> dbnet.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
>>> dbnet.initialize_initial_state()
>>> dbn_inf = DBNInference(dbnet)
>>> dbn_inf.query([('X', 0)], {('Y', 0):0, ('Y', 1):1, ('Y', 2):1})[('X', 0)].values
array([ 0.66594382,  0.33405618])
'''
