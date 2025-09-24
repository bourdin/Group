#%% Nonlinear viscosity experiment
# Find the viscosity which does not diverge.
# Through the experiments, C_nu = 0.01 was the smallest valuse
# which does not diverge.

from SWE import SWE
import numpy as np
cnu = 0.3

test_refsol = SWE()
test_refsol.set("t_end",2)
test_refsol.set("cnu",cnu)
test_refsol.set("eta_0",np.exp(-np.square(test_refsol.get('x')+1)/test_refsol.get('sigma')**2))
test_refsol.numerical(linearity = "non_linear_s")



def cnu_test(cnu):
    threshold = 2

    test = SWE()
    test.set("t_end",2)
    test.set("cnu",cnu)
    test.set("eta_0",np.exp(-np.square(test.get('x')+1)/test.get('sigma')**2))
    test.numerical(linearity = "non_linear_s")
    error = sum(abs(test.get("eta")-test_refsol.get("eta")))

    if error > threshold:
        test.plot_result()
        print("C_nu: ",cnu)
        print("Error: ",error)
        return 0
    else:
        test.plot_result()
        print("C_nu: ",cnu)
        print("Error: ",error)
        return 1

CnuList=list(range(7,12))
CnuList.reverse()

for i in CnuList:
    Test = cnu_test(i/100)
    if Test==1:
        print("Test passes -- Cnu = ",i/100)
    else:
        break
