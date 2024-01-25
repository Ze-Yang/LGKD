import sys
import csv

path = sys.argv[1]

mIoU_all_sum = 0.

bg = 0.
avg_old = 0.
avg_new = 0.
num_old_classes = 0
col_index = 0

with open(path, 'r') as f:
    data = list(csv.reader(f))
    num_rows = len(data)
    for step, x in enumerate(data):
        mIoU_all = float(x[-1])
        mIoU_all_sum += mIoU_all
        assert step == int(x[0]), 'Results for step {} missed.'.format(step)

        if step == 0:
            for col_index in range(1, len(x)):
                if x[col_index] in ('x', 'X'):
                    break
            num_old_classes = col_index - 2
            num_new_classes = len(x) - num_old_classes - 3
        elif step == num_rows - 1:
            if len(x[2:col_index]) == 0:
                avg_old = 0.
            else:
                avg_old = sum([float(i) for i in x[2:col_index]]) / num_old_classes
            if len(x[col_index:-1]) == 0:
                avg_new = 0.
            else:
                avg_new = sum([float(i) for i in x[col_index:-1]]) / num_new_classes
            bg = float(x[1])

print("{:<12}: {:>5d}".format("Final Step", step))
print("{:<12}: {:>2.2f}".format("IoU BG", 100 * bg))
print("{:<12}: {:>2.2f}".format("mIoU old", 100 * avg_old))
print("{:<12}: {:>2.2f}".format("mIoU new", 100 * avg_new))
print("{:<12}: {:>2.2f}".format("mIoU all", 100 * mIoU_all))
print("{:<12}: {:>2.2f}".format("mIoU Avg", 100 * mIoU_all_sum / num_rows))
