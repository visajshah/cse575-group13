# Importing the required libraries
import os
import subprocess

#All the required datafiles are uploaded on Dropbox.

# Function to download datafiles to local storage
def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def download_files():
    if os.path.isfile("combined_data_1.txt") and os.path.isfile("combined_data_2.txt") and os.path.isfile("combined_data_3.txt") and os.path.isfile("combined_data_4.txt"):
        print("Files already downloaded")
    else:
        print("Downloading files...")
        runcmd("wget -O combined_data_1.txt https://www.dropbox.com/s/hpkbyl6k16oo89j/combined_data_1.txt?dl=0", verbose = True)
        runcmd("wget -O combined_data_2.txt https://www.dropbox.com/s/cl3xfe1ex0syhki/combined_data_2.txt?dl=0", verbose = True)
        runcmd("wget -O combined_data_3.txt https://www.dropbox.com/s/cubavnzl8t7e6yh/combined_data_3.txt?dl=0", verbose = True)
        runcmd("wget -O combined_data_4.txt https://www.dropbox.com/s/l1cywln6m2vg9pp/combined_data_4.txt?dl=0", verbose = True)
        print("Finished downloading files")

# Function to create training data
# by merging data spread across 4 files.

def datafile():
    # Create a list of all datafile paths
    if os.path.isfile("data/data.csv"):
        print("Datafile already exists")
    else:
        datafiles = ["combined_data_{}.txt".format(x) for x in range(1, 5)]

        # We collect the merged data in a new file named train_data.csv
        data = open("data/data.csv", mode = 'w')
        
        for file in datafiles:
            print("Processing file {}/4 ...".format(datafiles.index(file) + 1))
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    if line[-1] != ':':
                        data.write(str(current_movie_id) + ',' + line + '\n')
                    else:
                        current_movie_id = int(line[:-1])
        data.close()
        print("Successfully created datafile")
