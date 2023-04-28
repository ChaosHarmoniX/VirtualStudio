import MHFormer.app.gen as gen

sc = gen.MHFormer()
import time
# 测时间
start = time.time()
sc.get_3D_kpt()
end = time.time()
print(end-start)
