import matplotlib.pyplot as plt
import json

def main():
    # file_list=['base40M-textvec-out.json','base40M-out.json','base300M-out.json','base1B-out.json']
    file_list=['base40M-textvec-out.json','base40M-out.json','base300M-out.json']
    label_dict={'base40M-textvec-out.json':'base40M, text-only','base40M-out.json':'base40M','base300M-out.json':'base300M'}
    for file_name in file_list:
        # Read the contents of the file
        with open(file_name, "r") as file:
            json_str = file.read()

        # Deserialize the JSON string to a dictionary
        temp_dict = json.loads(json_str)
        data=temp_dict['mem']
        
        timestamps = [t for t, _ in data]
        measured_val = [m for _, m in data]
        plt.plot(timestamps, measured_val,label=label_dict[file_name],linewidth=3)
    
    plt.scatter(46.63408136367798, 2891,c='black',s=20,zorder=10,marker='s')
    plt.scatter(44.72880721092224, 2891,c='black',s=20,zorder=10,marker='s')
    plt.scatter(139.16160225868225, 4957,c='black',s=20,zorder=10,marker='s')
    
    plt.scatter(59.079631090164185, 3215,c='black',s=20,zorder=10,marker='^')
    plt.scatter(67.021803855896, 3633,c='black',s=20,zorder=10,marker='^')
    plt.scatter(241.634019613266, 5841,c='black',s=20,zorder=10,marker='^')
    
    # x_point=42.38297986984253
    # y_point=2891
    # arrow_start = (x_point-5, y_point+500)  # Starting point of the arrow
    # arrow_end = (x_point, y_point)  # Ending point of the arrow

    # plt.arrow(*arrow_start,  # Unpacking the tuple into x and y components
    #         arrow_end[0] - arrow_start[0],  # Arrow length in x-direction
    #         arrow_end[1] - arrow_start[1],  # Arrow length in y-direction
    #         shape='right',  # Shape of the arrowhead
    #         color='red',  # Color of the arrow
    #         linewidth=1,  # Width of the arrow line
    #         length_includes_head=True,  # Include arrowhead in total length
    #         head_width=10,  # Width of the arrowhead
    #         head_length=10)  # Length of the arrowhead


    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12,loc='lower right')
    plt.grid(True)
    plt.savefig('./point-e-mem-plot.png')
    plt.show()

if __name__ == "__main__":
    main()