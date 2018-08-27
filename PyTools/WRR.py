def cmpt_wrr(b, o, s):
    return (b - s) / (b - o)

num_eval = int(raw_input("Number of evaluation sets\n"))
wrr = 0
for i in range(num_eval):
   b = float(raw_input("Baseline\n"))
   o = float(raw_input("Oracle\n"))
   s = float(raw_input("Semisup\n"))
   wrr += cmpt_wrr(b, o, s)
print 
print (wrr / num_eval)
