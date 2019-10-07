import numpy as np
import sys
first = True

if len(sys.argv) != 2:
    print('USAGE: python3 find_localmin.py [error_file]')
    exit(0)

neighbor = [
    [1, -1, 0],
    [1, 0, -1],
    [0, 1, -1],
    [0, -1, 1],
    [-1, 0, 1], 
    [-1, 1, 0]
]


with open(sys.argv[1], 'r') as f:
    
    f.readline()

    for i in range(3):
        
        vote_record = {}
        
        for j in range(9):

            record = np.zeros((11, 11, 11))
            
            for k in range(66):

                g, ss, sr, wb, wg, wr, error = f.readline().strip().split(',')
                
                index_b = int(float(wb)*10)
                index_g = int(float(wg)*10)
                index_r = int(float(wr)*10)

                record[index_b][index_g][index_r] = error

            for wb in range(11):
                for wg in range(10-wb, -1, -1):

                    wr = 10 - wb - wg

                    min_error = -1

                    for n in neighbor:

                        n_wb = wb + n[0]
                        n_wg = wg + n[1]
                        n_wr = wr + n[2]

                        if n_wb < 0 or n_wb > 10 or n_wg < 0 or n_wg > 10 or n_wr < 0 or n_wr > 10:
                            continue
                        elif min_error < 0 or record[n_wb][n_wg][n_wr] < min_error:
                            min_error = record[n_wb][n_wg][n_wr]

                    if record[wb][wg][wr] <= min_error:

                        key = '{}_{}_{}'.format(wb, wg, wr)
                        
                        if key in vote_record:
                            vote_record[key] += 1
                        else:
                            vote_record[key] = 1

        # print(vote_record) 
        keys = list(vote_record.keys())
        keys.sort(key=lambda x: vote_record[x], reverse=True)
        print([(i, vote_record[i]) for i in keys])