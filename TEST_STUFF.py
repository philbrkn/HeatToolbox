
import cma
import os
import matplotlib.pyplot as plt

cma_log_dir = "logs/test_hpc_mpi/cma_logs"
cma.plot(os.path.join(cma_log_dir, "outcma_"))
plt.show()
cma.s.figsave('fig325.png')
