
# in VS Code
# execute code with Alt+Enter
# run with Ctrl+F5

# import csv file created in matlab and saved for testing purposes
using CSV
using DataFrames

# test asr_calibrate
function test_asr()
    # example data stored in data.csv
        filename = "data.csv"
        filepath = joinpath(@__DIR__, filename)
        println(filepath)
    # import data
        println("reading file..")
        X = CSV.read(filepath, DataFrame, header = false)
    # and convert to array
        X = convert(Array, X)

    # fix this value for now
        srate = 250
        T = asr_calibrate(X, srate)
end

# start debugger
# Juno.@enter fit_eeg_distribution(X, min_clean_fraction, max_dropout_fraction)
