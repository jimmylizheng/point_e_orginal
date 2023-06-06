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
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('./point-e-mem-plot.png')
    plt.show()

if __name__ == "__main__":
    main()