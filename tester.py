from genericpath import isfile
import os

#TODO: convert to os.walk to make it more generalisable

'''tool to generate test commands when running large sets of experiments'''

def test_gen(path):
    return f"python main.py --test --run_folder {path} --num_testing_cases_repeat 4"

def test_not_done(path):
    L=os.listdir(path)
    for elem in L:
        if "test" in elem:
            group=os.listdir(path+'/'+elem)
            for elem in group:
                if "299" in elem:
                    return False
    return True

'''lets you filter by folder keyword if you only want to generate tests for a 
    specific one '''
def print_tests(path,folderKeyword="",retestExisting=False):
    if os.path.isfile(path):
        print("error")
        return
    else:
        for filename in os.listdir(path):
            if folderKeyword in filename:
                for subfilename in os.listdir(path+'/'+filename):
                    if (not os.path.isfile(path+'/'+filename+'/'+subfilename)) and (retestExisting or test_not_done(path+'/'+filename+'/'+subfilename)):
                        print(test_gen(path+'/'+filename+'/'+subfilename))
        return

# print(test_not_done("runs/multiple_networks/2022-06-15_22-33-56_giph"))
print_tests("runs/single_network",'dim')



