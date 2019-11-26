# test asr_calibrate

# import csv file created in matlab and saved for testing purposes
using CSV
using DataFrames

X = CSV.read("./data.csv"; header=false)
X = convert(Array, X)

srate = 250
