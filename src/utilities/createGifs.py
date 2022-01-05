
# read the testing files
# read the ranges file
# loop over the lap values and create graph for each one

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os
import pandas as pd
from sklearn import metrics

cwd=os.path.dirname(os.path.abspath(__file__))

test_y_path = os.path.join(cwd,"..",'..',"data","testing",  "testing_y_one.csv")
true_y_path=os.path.join(cwd,"..",'..',"data", "testing",  "actual_y_one.csv")
predict_y_path=os.path.join(cwd,"..",'..',"data", "testing",  "predicted_y_one.csv")
lab_ranges_path = os.path.join(cwd,"..",'..',"data", "lab_values", "lab_ranges.csv")



plt.style.use('seaborn-pastel')
fig = plt.figure(figsize=(8,6), dpi=150)
ax1 = plt.axes()
line, = ax1.plot([], [], lw=2)
plt.xlabel('Time (h)')
plt.ylabel('Value')

line1, =ax1.plot([],[], 'navy', label="Value Lower Limit")
line2, =ax1.plot([],[], 'coral',label="Value Upper Limit")
line3, =ax1.plot([], [], 'blue', label="True Lab Value")
line4 =ax1.scatter([],[], c='green',label="Predicted Normal",marker='o')
line5 =ax1.scatter([],[], c='red', label="Predicted OOR",marker='x')
lines = [line1,line2,line3,line4,line5]

init_zeros=np.zeros(5)



def mainGifs(icustayid):
    global y_true_f,results_pred_f,testing_Y_f,lab_low, lab_high,seq_length
    label_list = ["l_albumin", "l_arterial_be", "l_arterial_ph", "l_bun", "l_calcium", "l_chloride", "l_co2",
                  "l_creatinine", "l_glucose", "l_hb", "l_hco3", "l_inr", "l_ionised_ca", "l_lactate",
                  "l_paco2", "l_pao2", "l_platelets_count", "l_sodium", "l_total_bili", "l_wbccount",'l_spo2']


    results_pred=pd.read_csv(predict_y_path,names=label_list,index_col=False)
    results_pred.drop(0,axis=0,inplace=True)
    y_true=pd.read_csv(true_y_path)
    testing_Y=pd.read_csv(test_y_path)
    ranges=pd.read_csv(lab_ranges_path)

    seq_length = results_pred.shape[0]


    for f in label_list:

        results_pred_f = results_pred[f]
        y_true_f = y_true[f].values
        testing_Y_f = testing_Y[f]
        lab_low, lab_high = ranges[f]

        get_data(f)
        anim=plot_fn()
        file_name =os.path.join(cwd,'..','..',"data", "gifs", "{}_{}.gif".format(icustayid, f))
        print("Lab value {} graph saved".format(f))
        anim.save(file_name, writer='imagemagick', fps=1)


def init():
    line1.set_data(x_coordinates, limit_low)
    line2.set_data(x_coordinates, limit_high)
    line3.set_data([],[])
    init_zeros = np.zeros(2)
    line4.set_offsets(np.hstack((init_zeros[:1, np.newaxis], init_zeros[:1, np.newaxis])))
    line5.set_offsets(np.hstack((init_zeros[:1, np.newaxis], init_zeros[:1, np.newaxis])))
    #acc_text.set_text('initial')
    return tuple(lines)+(acc_text,)


def get_data(feature):

    global x_coordinates,acc_text,limit_low,limit_high,results_pred_norm_y,results_pred_out_y,results_pred_norm,results_pred_out

    plt.title('Lab value: {} '.format(feature))



    # set the x coordinates
    x_coordinates = np.arange(0, seq_length ) * 4

    #print("xcoord", x_coordinates)
    # set y axix:
    min_y=np.min(y_true_f)
    max_y=np.max(y_true_f)

    min_level=np.min([min_y,lab_low])-(0.05*np.min([min_y,lab_low]))
    max_level=np.max([max_y,lab_high])+(0.05*np.max([max_y,lab_high]))
    acc_text = ax1.text(seq_length*2-6, max_level*0.98, '', fontsize=8)
    #print("min and max level: ",min_level,max_level)
    ax1.set_ylim(min_level, max_level)

    # creates the 2 limit lines
    limit_low = np.repeat(lab_low, seq_length)
    limit_high = np.repeat(lab_high, seq_length)
    # set up the predictions lines on the graph
    results_pred_norm = np.array(np.where(results_pred_f == 0))
    results_pred_out = np.array(np.where(results_pred_f == 1))


    lab_low_pred = np.min([min_y,lab_low])-(0.02*np.min([min_y,lab_low]))
    lab_high_pred =np.min([min_y,lab_low])-(0.02*np.min([min_y,lab_low]))

    results_pred_norm_y = np.repeat(lab_low_pred, results_pred_norm.size)
    results_pred_out_y = np.repeat(lab_high_pred, results_pred_out.size)



def animate(i):

    if i in results_pred_out[0, :]:
        #print("{}th value = {}".format(i,"out of range"))
        result_out=np.where(results_pred_out[0, :] == i)
        result_out=result_out[0][0]
        x_coord_oor=results_pred_out[0, :result_out + 1] * 4
        data_2 = np.hstack((x_coord_oor[:, np.newaxis], results_pred_out_y[:result_out+1, np.newaxis]))
        line5.set_offsets(data_2)

    elif i in results_pred_norm[0,:]:
        #print("{}th value = {}".format(i,"normal"))
        result_normal=np.where(results_pred_norm[0, :] == i)
        result_normal=result_normal[0][0]
        x_coord_normal = results_pred_norm[0, :result_normal + 1] * 4
        data_1 = np.hstack((x_coord_normal[:, np.newaxis], results_pred_norm_y[:result_normal+1, np.newaxis]))
        line4.set_offsets(data_1)

    line3.set_data(x_coordinates[:i+1],y_true_f[:i+1])

    accuracy=metrics.accuracy_score(y_true=testing_Y_f[:i+1],y_pred=results_pred_f[:i+1])
    #print("accuracy score of the {}th element = {}".format(i,accuracy))
    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.

def plot_fn():

    ax1.set_xlim(0, seq_length*4)
    plt.xticks(np.arange(4, seq_length * 4, 4))
    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1000,repeat=True,frames=seq_length,blit=True)
    plt.legend(prop={"size":6})
    return anim



if __name__ == "__main__":

    stay_id="200025"
    mainGifs(stay_id)

