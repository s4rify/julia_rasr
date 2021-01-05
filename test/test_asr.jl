
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
        # returns (T,M,B,A)
        cal_state = asr_calibrate(X, srate)
        T = cal_state[1]
        M = cal_state[2]
        B = cal_state[3]
        A = cal_state[4]
        #TODO how to define tuples with named entries to call cal_state.T
end

# start debugger
# Juno.@enter fit_eeg_distribution(X, min_clean_fraction, max_dropout_fraction)
