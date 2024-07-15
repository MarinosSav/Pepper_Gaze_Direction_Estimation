import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

"""Global Variables"""
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
frame_height = 480
frame_width = 640

# Additional information that has been collected for each participant
extra_info = pd.DataFrame(np.array([['1', 169, 'Dark Brown', 'None'], ['2', 174, 'Blue', 'None'],
                                    ['3', 175, 'Blue', 'None'], ['4', 178, 'Green', 'None'],
                                    ['5', 188, 'Dark Brown', 'None'], ['6', 172, 'Brown', 'Glasses and Mask'],
                                    ['7', 170, 'Dark Brown', 'None'], ['8', 172, 'Dark Brown', 'None'],
                                    ['9', 168, 'Hazel', 'None'], ['10', 175, 'Brown', 'None']]),
                          columns=['ID', 'Height', 'Eye Color', 'Notes'])


def prepare(df, i, d):
    """Cleans and prepares the data for analysis. It takes as input the dataframe (df), the test ID (i) and the
    distance at which the test was performed (d) and outputs two dataframes. The first dataframe (df_res) returns
    information about the means of groups of tests, additional information on accuracy as well as extra information on
    each participant, whereas the second dataframe (df) returns a cleaner version of the dataframe inputted"""

    df['ID'] = str(i)
    # Attempt to fix the camera Y-offset caused by the experiment setup
    df['EstimatedY'] = df['EstimatedY'] + (d * math.tan(math.radians(72 / math.pi))) - \
                       (pd.merge(df, extra_info, on='ID')['Height'].astype(float) - 121)

    rows = df.groupby(['ID', 'S.', 'ExpX', 'ExpY'], as_index=False).count()['EstimatedX']

    # Remove extreme outliers (bigger than 2 standard deviations)
    mean = df.groupby(['ID', 'S.', 'ExpX', 'ExpY'])['EstimatedX'].transform('mean')
    std = df.groupby(['ID', 'S.', 'ExpX', 'ExpY'])['EstimatedX'].transform('std')
    not_outliers = df['EstimatedX'].between(mean.sub(std.mul(2)), mean.add(std.mul(2)), inclusive=False)
    df = df.loc[not_outliers]
    mean = df.groupby(['ID', 'S.', 'ExpX', 'ExpY'])['EstimatedY'].transform('mean')
    std = df.groupby(['ID', 'S.', 'ExpX', 'ExpY'])['EstimatedY'].transform('std')
    not_outliers = df['EstimatedY'].between(mean.sub(std.mul(2)), mean.add(std.mul(2)), inclusive=False)
    df = df.loc[not_outliers]

    new_rows = df.groupby(['ID', 'S.', 'ExpX', 'ExpY'], as_index=False).count()['EstimatedX']

    # Group tests first for each participant, then by the stage of the experiment and then by each point tested
    df_res = df.groupby(['ID', 'S.', 'ExpX', 'ExpY'], as_index=False)[
        'EstimatedX', 'EstimatedY', 'EyeX', 'EyeY', 'HeadX', 'HeadY'].mean()
    # Calculate accuracy and errors in estimations
    df_res = df_res.assign(AccuracyX=(100 - ((abs(df_res.ExpX - df_res.EstimatedX) / df_res.ExpX) * 100)),
                           AccuracyY=(100 - ((abs(df_res.ExpY - df_res.EstimatedY) / df_res.ExpY) * 100)),
                           ErrorX=abs(df_res.ExpX - df_res.EstimatedX),
                           ErrorY=abs(df_res.ExpY - df_res.EstimatedY),
                           ErrorE=((((df_res.EstimatedX - df_res.ExpX) ** 2) +
                                    ((df_res.EstimatedY - df_res.ExpY) ** 2)) ** 0.5))
    df_res['Uncertainty'] = ((rows - new_rows) / rows) * 100
    df_res['Uncertainty'] = df_res['Uncertainty'].round(1)

    # Combine the participant information
    df_res = pd.merge(df_res, extra_info, on='ID')

    return df_res, df


accuracy = []
plot_data = []
# Iterate through every single participant and store information on tests
for i in range(1, 11):
    if i == 1:
        df80, df_plot80 = prepare(pd.read_pickle("test_" + str(i) + "_80"), i, 80)
        df120, df_plot120 = prepare(pd.read_pickle("test_" + str(i) + "_120"), i, 120)
    else:
        df_res, df_plot80 = prepare(pd.read_pickle("test_" + str(i) + "_80"), i, 80)
        df80 = pd.concat([df80, df_res], ignore_index=True)
        df_res, df_plot120 = prepare(pd.read_pickle("test_" + str(i) + "_120"), i, 120)
        df120 = pd.concat([df120, df_res], ignore_index=True)
    plot_data.append([i, df_plot80, df_plot120])
    accuracy.append([i, int(df80['AccuracyX'].mean()), int(df80['AccuracyY'].mean()),
                     int(df120['AccuracyX'].mean()), int(df120['AccuracyY'].mean()),
                     round(df80['ErrorE'].mean(), 2), round(df120['ErrorE'].mean(), 2),
                     round(df80['ErrorX'].mean(), 2), round(df120['ErrorX'].mean(), 2),
                     round(df80['ErrorY'].mean(), 2), round(df120['ErrorY'].mean(), 2)])

# Print all cleaned data (useful for more in-depth analysis)
print("\n\n80cm:\n", df80, "\n\n120cm:\n", df120, "\n\n", extra_info.to_string(index=False))

# Print a summary of each participant
for row in accuracy:
    print("\n Test " + str(row[0]) + " at distance 80cm  >>  X-Coordinate Accuracy: " + str(row[1]) +
          "% | Y-Coordinate Accuracy: " + str(row[2]) + "% | Mean Euclidean Distance Error: " + str(row[5]) +
          " px (X Error: " + str(row[7]) + "px | Y Error: " + str(row[9]) + "px)\n Test " + str(row[0]) +
          " at distance 120cm >>  X-Coordinate Accuracy: " + str(row[3]) +
          "% | Y-Coordinate Accuracy: " + str(row[4]) + "% | Mean Euclidean Distance Error: " + str(row[6]) +
          " px (X Error: " + str(row[8]) + "px | Y Error: " + str(row[10]) + "px)")

print("\n------------------------------------------------------------------------------------------------------------")
# Print a summary on the performance of the algorithm
accuracy = np.array(accuracy)
print("\n Overall at distance 80cm  >>  X-Coordinate Accuracy: " + str(round(float(np.mean(accuracy[:, 1])), 2)) +
      "% | Y-Coordinate Accuracy: " + str(round(float(np.mean(accuracy[:, 2])), 2)) +
      "% | Mean Euclidean Distance Error: " + str(round(float(np.mean(accuracy[:, 5])), 2)) + " px (X Error: " +
      str(round(float(np.mean(accuracy[:, 7])), 2)) + "px | Y Error: " + str(round(float(np.mean(accuracy[:, 9])), 2)) +
      "px)\n Overall at distance 120cm >>  X-Coordinate Accuracy: " + str(round(float(np.mean(accuracy[:, 3])), 2)) +
      "% | Y-Coordinate Accuracy: " + str(round(float(np.mean(accuracy[:, 4])), 2)) +
      "% | Mean Euclidean Distance Error: " + str(round(float(np.mean(accuracy[:, 6])), 2)) + " px (X Error: " +
      str(round(float(np.mean(accuracy[:, 8])), 2)) + "px | Y Error: " +
      str(round(float(np.mean(accuracy[:, 10])), 2)) + "px)")

# Visualize data
grid_width = 4
grid_height = 3
k = 0

fig80, ax80 = plt.subplots(grid_height, grid_width, sharex='col', sharey='row')
fig120, ax120 = plt.subplots(grid_height, grid_width, sharex='col', sharey='row')
fig80.subplots_adjust(hspace=0.5, wspace=0.5)
fig120.subplots_adjust(hspace=0.5, wspace=0.5)
for i in range(grid_height):
    for j in range(grid_width):
        ax80[i, j].set_xlim([0 - (frame_width / 2), frame_width + (frame_width / 2)])
        ax80[i, j].set_ylim([0 - (frame_height / 2), frame_height + (frame_height / 2)])
        plot_data[k][1].plot(kind='scatter', x='EstimatedX', y='EstimatedY', color='red', ax=ax80[i, j], s=[10])
        plot_data[k][1].plot(kind='scatter', x='ExpX', y='ExpY', color='blue', ax=ax80[i, j], s=[20], marker="x")
        df80[df80['ID'] == str(k + 1)].plot(kind='scatter', x='EstimatedX', y='EstimatedY', color='green',
                                            ax=ax80[i, j], s=[20])

        ax120[i, j].set_xlim([0 - (frame_width / 2), frame_width + (frame_width / 2)])
        ax120[i, j].set_ylim([0 - (frame_height / 2), frame_height + (frame_height / 2)])
        plot_data[k][2].plot(kind='scatter', x='EstimatedX', y='EstimatedY', color='red', ax=ax120[i, j], s=[10])
        plot_data[k][2].plot(kind='scatter', x='ExpX', y='ExpY', color='blue', ax=ax120[i, j],  s=[20], marker="x")
        df120[df120['ID'] == str(k + 1)].plot(kind='scatter', x='EstimatedX', y='EstimatedY', color='green',
                                              ax=ax120[i, j],  s=[20])

        k += 1
        if k == len(plot_data):
            break

plt.show()
