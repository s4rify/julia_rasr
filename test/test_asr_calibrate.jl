# test asr_calibrate

# import csv file created in matlab and saved for testing purposes
using CSV
using DataFrames

# example data stored in data.csv
filename = "data.csv"
filepath = joinpath(@__DIR__, filename)
println(filepath)
# import data
println("reading file..")
X = CSV.read(filepath; header=false)
# and convert to array
X = convert(Array, X)

# fix this value for now
srate = 250
