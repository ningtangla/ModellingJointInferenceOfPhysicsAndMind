import time
from subprocess import Popen, PIPE


def generateTrajectory(codeName, numSample):
    sampleIdStrings = list(map(str, range(numSample)))
    cmdList = [['python3', codeName, sampleIndex] for sampleIndex in sampleIdStrings]
    processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
    for proc in processList:
        stdOut, stdErr = proc.communicate()
        proc.wait()
    return cmdList

def main():
    codeName = 'prepareNeuralNetData.py'
    numSample = 40
    stratTime = time.time()
    cmdList = generateTrajectory(codeName, numSample)

    processTime = time.time() - stratTime
    print('subProcess', processTime)

if __name__ == "__main__":
    main()
